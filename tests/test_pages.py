"""The pieces every rebuilt page is made of: a fact and its order, the words beside the main button, where a value came
from, the form with its rows, its fold and its name and note, the foot, "In use", and the page that holds them, with
when it reads and how it hosts a run. They are read here on a stand-in page as awkward as a real one
(``tests/_fixtures.build_step_page``), in a whole window where a page's place in it matters.
"""
import itertools
import re
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QFont, QFontInfo
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QScrollArea, QWidget

from core.gui import design, fields, fonts, style
from core.gui.panels.inference.config_tab import ConfigPanel
from core.gui.panels.inference.posterior_tab import PosteriorPanel
from core.gui.panels.inference.prior_tab import PriorPanel
from core.gui.run.controller import LIVE_TEXT, NOT_READ_YET_TEXT, controller
from core.gui.step_facts import (LEADS, REFUSED_BLOCKS, Fact, about_words, ordered, origin_words, reading_words,
                                  reason_beside, refused_facts)
from core.gui.widgets.choice import Choice
from core.gui.widgets.facts import FACTS_CAP, Facts, Foot
from core.gui.widgets.form import FOLD_NOT_USUAL, FOLD_WORDS, NAME_FREE_TEXT, Form, NameAndNote, TickedField
from core.gui.widgets.in_use import InUse
from core.gui.widgets.labeled_inputs import COMMA_SENTENCE, FloatField, IntField
from core.gui.widgets.message_row import MessageRow
from tests._dialogs import all_asked, asked
from tests._fixtures import (TWO_LINES, build_part, build_run_host, build_step_page, build_window, qt_app,
                             readings_of_a_run)
from tests._look_helpers import _gone, _settle, _the_look, text_size
from tests._runs import Clock, StandIn, at, controller_on, start_run, turn_until
from tests._window_helpers import in_front, shown_off_screen

BLOCKS = "the name cannot be used"
#: Each rebuilt page by its step, as the standing tests a page joins in its own task name it: the stand-in, then the
#: pages of the window.
PAGES = {"the stand-in": None, "Model and options": ConfigPanel, "Prior": PriorPanel, "Training": PosteriorPanel}
#: A field of each page an edit is typed into, by the page's class.
EDITED = {None: "num_runs", ConfigPanel: "chi_n_freqs", PriorPanel: "num_iterations", PosteriorPanel: "num_runs"}
#: A record's name of the length the store lets stand, which no line of an origin holds at the window's least width.
LONG_NAME = "retrain_narrowing_round_of_the_second_of_november"


@pytest.fixture(autouse=True)
def _the_programs_font_put_back():
    """``_a_page`` stands a page in the program's own font: a part measured in another is measured in vain. What a test
    finds of the fonts, the next test finds too: the application's, each class's, and what a font that names no face is."""
    app = qt_app()
    fonts.register_fonts()                   # once per process: the first test to ask for a font is no fault of its
    kept = QFont(app.font())

    def held():
        return (app.font().toString(), tuple(QApplication.font(name).toString() for name in fonts._CLASSES),
                QFontInfo(QFont()).family())

    found = held()
    yield
    app.setFont(kept)
    assert held() == found, "the program's font stayed after the test"


def _a_page(tmp_path, *, clock=None, shown=True, size=(1300, 820), programs_font=True, cls=None):
    """The stand-in page, or ``cls``, in a whole window on the test's own clock, in the program's own font at the text size
    in force, in sight at ``size`` when ``shown``, with what is read as it comes into sight forgotten: (window, page, the
    clock)."""
    app = qt_app()
    if programs_font:
        fonts.load_app_font(app)
    clock = clock or Clock(at(8, 12, 4))
    controller_on(tmp_path, clock)
    window = build_window()
    design.set_room((3400, 1380))
    window.setMinimumSize(1, 1)
    page = build_step_page(cls, window=window)
    if shown:
        shown_off_screen(window, *size)
        window.go_to("inference", "training")
        _settle()
    if page.names is not None:
        page.names.name.setText("retrain_3")
        page.names.check_now()
    page.readings = 0
    return window, page, clock


def _start(page, stand_in, **kwargs) -> None:
    """A stand-in run started through the page's own ``dispatch``, at the page's own place."""
    from core.gui.run.model import RunSpec
    start_run(page, stand_in, spec=kwargs.pop("spec", None) or RunSpec(stand_in.word, ("inference", page.step),
                                                                       stop_button="Stop the run…"), **kwargs)


def _edge_of(widget, window) -> QRect:
    """``widget`` in the window's own coordinates."""
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def _in_view(page, widget) -> bool:
    """Whether the whole of ``widget``, in the page's body, lies in the viewport, with the room over and under it for its
    focus ring."""
    port = page.scroller.viewport()
    room = design.SIZE["focus_ring"] + design.SIZE["focus_gap"]
    return port.rect().contains(_edge_of(widget, port).adjusted(0, -room, 0, room))


def _walked(window, count) -> list:
    """From the control that holds the keyboard, what it stands on over ``count`` stops, with real keys."""
    from PySide6.QtTest import QTest
    walked = []
    for _ in range(count):
        walked.append(QApplication.focusWidget())
        QTest.keyClick(walked[-1], Qt.Key_Tab)
        _settle()
    return walked


def _the_form():
    qt_app()
    return build_part(Form)


# ── a fact, as values ──
def test_every_error_and_blocked_fact_blocks_and_nothing_else_does():
    """A marker never disagrees with the button: an "Error." and a "Blocked." hold the words that go beside it and no other
    fact does, and a fact that says otherwise cannot be made. Words of nothing but spaces are no words. Blocking facts come
    first, each group in the order handed in."""
    for kind in ("err", "lock"):
        assert Fact(kind, "A sentence.", BLOCKS).blocks == BLOCKS
        for words in ("", "   "):
            with pytest.raises(ValueError):
                Fact(kind, "A sentence.", words)
    for kind in ("warn", "info", "fine"):
        assert Fact(kind, "A sentence.").blocks == ""
        with pytest.raises(ValueError):
            Fact(kind, "A sentence.", BLOCKS)
    with pytest.raises(ValueError):
        Fact("done", "A marker of the rail is no fact's.")
    assert LEADS == {"err": "Error.", "lock": "Blocked.", "warn": "Attention.", "info": "Information.", "fine": "Fine."}
    fine, info, one, two = Fact("fine", "a"), Fact("info", "b"), Fact("lock", "c", "x"), Fact("err", "d", "y")
    assert ordered([fine, one, info, two]) == (one, two, fine, info)


@pytest.mark.parametrize("facts, live, nothing, words", [
    ((), False, False, ""),
    ((), False, True, "Nothing to apply: these choices are in use."),
    ((Fact("err", "a", BLOCKS),), False, False, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS), Fact("lock", "b", "needs a prior")), False, False,
     "Cannot start yet: the name cannot be used, and 1 more above."),
    ((Fact("err", "a", BLOCKS), Fact("lock", "b", "needs a prior"), Fact("err", "c", "z")), False, False,
     "Cannot start yet: the name cannot be used, and 2 more above."),
    ((Fact("fine", "a"), Fact("err", "b", BLOCKS)), False, False, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS),), False, True, "Cannot start yet: the name cannot be used."),
    ((Fact("err", "a", BLOCKS),), True, True, "A run is live. Only one runs at a time."),
    ((), True, False, "A run is live. Only one runs at a time."),
], ids=range(9))
def test_the_reason_beside_the_button_for_each_state(facts, live, nothing, words):
    """The table of what stands beside a greyed button, exact, a live run first and what blocks before nothing to apply.
    The live words are the ones a page says when it asks for a run while one is live."""
    assert reason_beside(facts, live=live, nothing_to_apply=nothing) == words
    assert reason_beside((), live=True, nothing_to_apply=False) == LIVE_TEXT


def test_the_reading_line_says_when_the_records_were_read():
    """The three sentences, the time from the moment handed in, and nothing for a page not read with no run live: it reads
    as it is shown."""
    moment = at(8, 9, 4)
    assert reading_words(moment, live=False) == "The saved records were read at 09:04."
    assert reading_words(moment, live=True) == "A run is live. Read at 09:04; read again when it ends."
    assert reading_words(None, live=True) == NOT_READ_YET_TEXT == "A run is live. Not read yet; read when it ends."
    assert reading_words(None, live=False) == ""


SOURCES = {"plan": ("retrain", "from plan retrain", "changed from the plan"),
           "record": ("retrain_posterior", "from record retrain_posterior", "changed from the record"),
           "run": ("retrain_3", "from run retrain_3", "changed from the run")}


def test_each_field_has_exactly_one_origin_over_every_combination():
    """Where the value came from, over every combination of what can be said of a field: where the source in use holds the
    setting, its two rows win; else a value put back and not touched since is kept from last time, whether or not it
    equals the usual; else the usual or a change from it. Every combination has one answer, and each of the table's
    seven words stands."""
    seen = set()
    for held, at_source, at_usual, kept, source in itertools.product((True, False), (True, False), (True, False),
                                                                     (True, False), SOURCES):
        name, unchanged, changed = SOURCES[source]
        words = origin_words(held=held, at_source=at_source, at_usual=at_usual, kept=kept, source=source, name=name)
        want = ((unchanged if at_source else changed) if held else "kept from last time" if kept
                else "usual" if at_usual else "changed from the usual")
        assert words == want, (held, at_source, at_usual, kept, source)
        seen.add(words)
    assert seen == {"usual", "changed from the usual", "kept from last time", *(w for s in SOURCES.values() for w in s[1:])}
    assert origin_words(held=True, at_source=True, at_usual=False, kept=False) == "from plan", "a source with no name"


def test_a_refusal_among_the_facts_is_an_error_with_the_words_every_refusal_has():
    assert refused_facts(("One.", "Two.")) == (Fact("err", "One.", REFUSED_BLOCKS), Fact("err", "Two.", REFUSED_BLOCKS))
    assert refused_facts(()) == () and REFUSED_BLOCKS == "the last start was refused"


@pytest.mark.parametrize("seconds, words", [
    (1, "about half a minute"), (15, "about half a minute"), (30, "about half a minute"), (44, "about half a minute"),
    (45, "about a minute"), (60, "about a minute"), (75, "about a minute and a half"), (92, "about a minute and a half"),
    (120, "about 2 minutes"), (150, "about 2 and a half minutes"), (570, "about 9 and a half minutes"),
    (585, "about 10 minutes"), (600, "about 10 minutes"), (629, "about 10 minutes"), (631, "about 11 minutes"),
    (3540, "about 59 minutes"), (3570, "about an hour"), (3599, "about an hour"), (3600, "about an hour"),
    (5400, "about an hour and a half"), (7200, "about 2 hours"), (8999, "about 2 and a half hours"),
    (9000, "about 2 and a half hours"), (90_000, "about 25 hours"), (0, ""), (-3.0, ""), (float("nan"), ""),
    (float("inf"), "")])
def test_about_words_round_a_length_as_a_person_would(seconds, words):
    """To the half minute below ten minutes, to the minute below an hour, to the half hour above, a half going up as a
    person rounds, and an hour never "60 minutes". A length that is no length says nothing."""
    assert about_words(seconds) == words


# ── a row ──
def test_a_row_stands_label_field_and_unit_and_its_label_is_its_fields_buddy():
    """The form names the row by its field key, keeps it, and reads the field as a number: the commas it shows are not
    read, and what it is told it shows grouped."""
    form = _the_form()
    field = IntField("5000", grouped=True)
    row = form.add("num_runs", field, "batches", usual=5000)
    assert form.rows["num_runs"] is row and (row.key, row.widget) == ("num_runs", field)
    assert row.label_text() == fields.label("num_runs") and row.label.buddy() is field
    assert row.unit.text() == "batches" and not row.unit.isHidden()
    assert row.value() == 5000 and row.usual() == 5000 and field.text() == "5,000"
    row.set_value(10000)
    assert field.text() == "10,000" and row.value() == 10000
    row.set_origin("usual")
    assert row.origin.text() == "usual"
    assert form.add("run_size_cap", IntField(4), usual=4).unit.isHidden(), "a row with no unit keeps no room for one"
    assert form.add("chi_mode", Choice(), label="Kind of observation").label_text() == "Kind of observation"


def test_a_row_named_nowhere_in_the_table_fails_as_it_is_built():
    with pytest.raises(KeyError):
        _the_form().add("no_such_key", IntField(1))


def test_a_label_runs_through_the_tables_rich_form():
    """A label the table writes with a subscript shows it, and reads back as the table wrote it."""
    form = _the_form()
    row = form.add("t_obs", FloatField(4.5))
    assert row.label_text() == "T_obs (s)" and row.label.text() != "T_obs (s)"


def test_a_number_field_stands_right_aligned_and_120_wide():
    """A number stands as the look stands it, in a ticked field too, at every text size."""
    with text_size(1.5):
        form = _the_form()
        plain, inner = FloatField(1.5), IntField(0)
        form.add("learning_rate", plain, usual=1.5)
        form.add("run_size_cap", TickedField(inner, "Automatic", 0), usual=0)
        for field in (plain, inner):
            assert field.alignment() & Qt.AlignRight, type(field)
            assert (field.minimumWidth(), field.maximumWidth()) == (180, 180), "number_w, at the text size in force"
            assert field.font().featureValue(field.font().Tag.fromString("tnum")) == 1, "digits of equal width"


def test_a_value_the_window_may_not_change_is_text_with_its_origin():
    form = _the_form()
    row = form.add_text("chi_f0", "Drive amplitude in training: 0.15 (no unit).", "fixed by measurement")
    assert row.widget is None and row.origin.text() == "fixed by measurement"
    assert row.value() == row.label_text() == "Drive amplitude in training: 0.15 (no unit)."
    assert row.focus_target() is None and form.first_control() is None
    assert not form.findChildren(QLineEdit), "a value that cannot be changed is no field"
    row.set_value("Drive amplitude in training: 0.2 (no unit).")
    assert row.label.text() == "Drive amplitude in training: 0.2 (no unit)."


@pytest.mark.parametrize("build, typed, value, words", [
    (lambda: FloatField("0.0005"), 0.001, 0.001, "0.001"),
    (lambda: IntField(5, grouped=True), 12000, 12000, "12,000"),
    (lambda: IntField(None), None, None, "blank"),
], ids=["a float", "a grouped whole number", "a blank box"])
def test_a_row_reads_and_sets_a_number(build, typed, value, words):
    form = _the_form()
    row = form.add("learning_rate", build())
    row.set_value(typed)
    assert (row.value(), row.words()) == (value, words)


def test_a_row_reads_and_sets_a_list_by_its_items_data_or_else_its_words():
    """The value of a list is what its item stands for, a False among them; told a value it does not hold, it stays."""
    form = _the_form()
    kinds = Choice()
    kinds.addItem("Set by the bounds file", False)
    kinds.addItem("Frequency response", True)
    plain = Choice()
    plain.addItems(["On", "Off"])
    row, words = form.add("chi_mode", kinds, usual=False, label="Kind"), form.add("units", plain, usual="On")
    assert (row.value(), words.value(), row.at_usual()) == (False, "On", True)
    row.set_value(True)
    assert (row.value(), kinds.currentIndex(), row.at_usual(), row.words()) == (True, 1, False, "Frequency response")
    row.set_value(False)
    assert kinds.currentIndex() == 0
    words.set_value("Off")
    assert (words.value(), words.words()) == ("Off", "Off")
    words.set_value("Neither")
    assert words.value() == "Off"


def test_a_row_reads_and_sets_a_tick_box_and_a_line_of_text():
    from PySide6.QtWidgets import QCheckBox
    form = _the_form()
    tick, line = form.add("units", QCheckBox(), usual=True), form.add("name", QLineEdit("a"), label="Name", usual="a")
    assert (tick.value(), tick.at_usual(), tick.words()) == (False, False, "off")
    tick.set_value(True)
    assert (tick.value(), tick.at_usual(), tick.words()) == (True, True, "on")
    assert (line.value(), line.at_usual()) == ("a", True)
    line.set_value(None)
    assert (line.value(), line.words()) == ("", "blank")


def test_a_ticked_field_reads_its_ticked_value_and_greys_its_box():
    """Ticked, the field is greyed and keeps what it held, and the value is what the tick stands for; "No limit" and a blank
    box both read None, so that a caller reads the tick first. The box starts ticked when the field reads as the value it
    stands for. ``changed`` hears the tick and the field."""
    automatic = TickedField(IntField("0"), "Automatic", 0)
    assert automatic.ticked_box.isChecked() and not automatic.field.isEnabled() and automatic.value_or_none() == 0
    automatic.ticked_box.setChecked(False)
    assert automatic.field.isEnabled() and automatic.field.text() == "0"
    heard = []
    automatic.changed.connect(lambda: heard.append(1))
    automatic.field.setText("2048")
    assert automatic.value_or_none() == 2048 and heard == [1]
    automatic.ticked_box.setChecked(True)
    assert (automatic.value_or_none(), automatic.field.text(), not automatic.field.isEnabled()) == (0, "2048", True)
    assert heard == [1, 1]
    automatic.set_value(4096)
    assert (automatic.is_ticked(), automatic.value_or_none(), automatic.field.text()) == (False, 4096, "4096")
    automatic.set_value(0)
    assert automatic.is_ticked() and automatic.field.text() == "4096"
    limit = TickedField(IntField(None), "No limit", None)
    assert limit.is_ticked() and limit.value_or_none() is None
    limit.ticked_box.setChecked(False)
    assert limit.value_or_none() is None, "a blank box reads as no limit too: read the tick first"
    typed = TickedField(IntField(300), "No limit", None)
    assert not typed.is_ticked() and typed.value_or_none() == 300 and typed.words() == "300"
    typed.ticked_box.setChecked(True)
    assert typed.words() == "No limit"


def test_a_ticked_row_is_read_as_its_box_says_and_its_unit_counts_the_field():
    """Ticked, a blank field is nothing wrong; unticked it is; and the unit stands after the field it counts, not after the
    box. A value said of a ticked row is the box's words with no unit."""
    form = _the_form()
    ticked = TickedField(IntField(None), "No limit", None)
    row = form.add("max_num_epochs", ticked, "epochs", usual=None, label="Epoch limit")
    assert row.reading() is None and row.words() == "No limit"
    ticked.ticked_box.setChecked(False)
    assert row.reading() == "Epoch limit is blank."
    ticked.field.setText("300")
    assert row.reading() is None and row.words() == "300 epochs"
    layout = ticked.layout()
    assert [layout.itemAt(at).widget() for at in range(3)] == [ticked.field, row.unit, ticked.ticked_box]


def test_a_help_mark_stands_beside_the_label_and_no_key_stops_there():
    form = _the_form()
    row = form.add("learning_rate", FloatField(1.0), usual=1.0, help_text="How far a step goes.")
    assert row.badge.focusPolicy() == Qt.NoFocus and row.badge.toolTip() == "How far a step goes."
    assert form.add("num_runs", IntField(1)).badge is None


def test_the_line_under_a_field_is_one_sentence_with_its_marker():
    """One sentence under the field, with its marker, in the lettering face and as plain words: in error the field is
    outlined, and with the line taken away the outline goes. A second line replaces the first."""
    form = _the_form()
    field = IntField(1)
    row = form.add("num_runs", field)
    assert row.line.isHidden()
    row.set_line("err", "Batches must be at least 1 <here>.")
    assert (row.line.isHidden(), row.line.marker.kind, row.line.label.text()) == (
        False, "err", "Batches must be at least 1 <here>.")
    assert row.line.label.textFormat() == Qt.PlainText and field.property("state") == "error" and row.in_error()
    row.set_line("warn", "Changed, not applied. In use: master.txt.")
    assert (row.line.marker.kind, field.error(), field.property("state"), row.in_error()) == (
        "warn", "", "", False)
    row.set_line("", "")
    assert row.line.isHidden()
    line = QLineEdit()
    other = form.add("name", line, label="Name")
    other.set_line("err", "Taken.")
    assert line.property("state") == "error"
    other.set_line("", "")
    assert line.property("state") == ""


def test_a_fields_own_sentence_stands_under_it():
    """A comma says why under the field as soon as editing ends, and not only in the tool tip; mended, the line goes."""
    form = _the_form()
    field = FloatField("1.5")
    row = form.add("learning_rate", field)
    field.setText("1,5")
    field.editingFinished.emit()
    assert (row.line.label.text(), row.line.marker.kind, row.in_error()) == (COMMA_SENTENCE, "err", True)
    row.set_line("", "")
    assert (row.line.label.text(), row.line.isHidden()) == (COMMA_SENTENCE, False), "a page took the comma's sentence away"
    field.setText("1.5")
    assert row.line.isHidden() and not row.in_error()


def test_a_row_hides_whole_and_comes_back():
    """A row its model does not use takes no room and no key, its line, its unit and its sentence with it; and what it
    showed comes back with it, a unit it never had not among that."""
    form = _the_form()
    field = IntField(6)
    row = form.add("chi_n_freqs", field, "recordings")
    other = form.add("chi_k_pad", IntField(12))
    said = form.add_sentence("Slots: the most driven recordings one observation can hold.", under="chi_n_freqs")
    row.set_line("warn", "Changed, not applied.")
    row.set_shown(False)
    assert all(part.isHidden() for part in (row.label, field, row.unit, row.origin, row.line, said))
    row.set_line("err", "Said while it is hidden.")
    assert row.line.isHidden(), "a line said of a hidden row came into sight"
    row.set_line("warn", "Changed, not applied.")
    row.set_shown(True)
    assert not any(part.isHidden() for part in (row.label, field, row.unit, row.origin, row.line, said))
    other.set_shown(False)
    other.set_shown(True)
    assert other.unit.isHidden() and other.line.isHidden()
    typed = QLineEdit()
    row.add_under(typed)
    typed.setVisible(False)
    row.set_shown(False)
    row.set_shown(True)
    assert typed.isHidden(), "what a page keeps hidden under a row came back with it"


def test_a_row_says_where_the_keys_stop_what_stands_under_it_among_that():
    """The field, then the box of a ticked field, then what stands under the field; a value that is only shown is no stop."""
    form = _the_form()
    ticked = TickedField(IntField(1), "Automatic", 0)
    row = form.add("run_size_cap", ticked)
    typed = QLineEdit()
    row.add_under(typed)
    assert row.stops() == [ticked.field, ticked.ticked_box, typed]
    assert form.add_text("chi_f0", "Shown.", "fixed by measurement").stops() == []


# ── the fold ──
def _a_fold():
    """A form with main rows and three folded ones of three kinds; the fold, with the rows by name."""
    form = _the_form()
    form.add("num_runs", IntField(5000, grouped=True), usual=5000)
    rate = form.add("learning_rate", FloatField("0.0005"), usual=0.0005, folded=True)
    size = form.add("run_size_cap", TickedField(IntField("0"), "Automatic", 0), usual=0, folded=True)
    limit = form.add("max_num_epochs", TickedField(IntField(None), "No limit", None), "epochs", usual=None, folded=True,
                     label="Epoch limit")
    return form, rate, size, limit


def test_the_fold_counts_its_settings_and_those_not_usual():
    """The words of the button, the singular among them; a number of eleven is not the singular."""
    assert FOLD_WORDS.format(count="7 settings") == "Rarely changed (7 settings)"
    assert FOLD_NOT_USUAL.format(count="7 settings", m=2) == "Rarely changed (7 settings, 2 not usual)"
    form, rate, size, _limit = _a_fold()
    assert form.fold.button.text() == "Rarely changed (3 settings)"
    rate.set_value(0.001)
    assert form.fold.button.text() == "Rarely changed (3 settings, 1 not usual)"
    size.widget.ticked_box.setChecked(False)
    size.widget.field.setText("2048")
    assert form.fold.button.text() == "Rarely changed (3 settings, 2 not usual)"
    single = _the_form()
    single.add("learning_rate", FloatField(1.0), usual=1.0, folded=True)
    assert single.fold.button.text() == "Rarely changed (1 setting)"
    many = _the_form()
    for key in ("n_cal", "cal_n_scales", "num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
                "min_cluster_size", "min_samples", "hidden_features", "num_transforms"):
        many.add(key, IntField(1), usual=1, folded=True)
    assert many.fold.button.text() == "Rarely changed (11 settings)"


def test_not_usual_is_judged_as_the_field_reads_never_as_it_is_typed():
    """A number written another way is the same number; a blank box is not the usual value, and neither is a comma, whatever
    the digits; the tick for "No limit" is the usual value of a row whose usual value is none, and the box under it, left
    blank with the tick off, is not."""
    form, rate, _size, limit = _a_fold()
    assert (limit.at_usual(), form.fold.button.text()) == (True, "Rarely changed (3 settings)")
    for typed, usual in (("5e-4", True), ("0.00050", True), ("", False), ("0,0005", False), ("0.0006", False)):
        rate.widget.setText(typed)
        assert rate.at_usual() is usual, typed
    rate.widget.setText("0.0005")
    limit.widget.ticked_box.setChecked(False)
    assert not limit.at_usual() and limit.value() is None, "a blank box was read as the tick it reads alike"
    assert form.fold.button.text().endswith("1 not usual)") and form.fold.summary.text().startswith("Epoch limit blank")
    limit.widget.field.setText("300")
    assert not limit.at_usual() and form.fold.button.text().endswith("1 not usual)")
    whole = _the_form()
    assert whole.add("chi_max_cycles", FloatField(20.0), usual=20).at_usual(), "20.0 is 20"


def test_the_folds_sentence_names_those_not_usual_first():
    form, rate, _size, limit = _a_fold()
    assert form.fold.summary.text() == ("Learning rate 0.0005; Simulations per batch Automatic; "
                                        "Epoch limit No limit")
    limit.widget.ticked_box.setChecked(False)
    limit.widget.field.setText("300")
    assert form.fold.summary.text() == ("Epoch limit 300 epochs; Learning rate 0.0005; "
                                        "Simulations per batch Automatic")
    rate.widget.setText("")
    assert form.fold.summary.text().split("; ")[:2] == ["Learning rate blank", "Epoch limit 300 epochs"]


def test_the_fold_stands_shut_at_first_and_shows_its_rows_in_place_of_their_values():
    form, rate, _size, _limit = _a_fold()
    fold = form.fold
    assert not fold.is_open() and fold.body.isHidden() and not fold.summary.isHidden()
    fold.button.click()
    assert fold.is_open() and not fold.body.isHidden() and fold.summary.isHidden()
    fold.button.click()
    assert not fold.is_open() and fold.body.isHidden()


def test_a_folded_field_in_error_opens_the_fold_and_keeps_it_open():
    """It opens by itself and cannot be shut until the error has gone, whichever way the error came; and the error going
    leaves it open for the person to shut. An error among the main settings opens nothing."""
    form, rate, _size, _limit = _a_fold()
    fold = form.fold
    rate.set_line("err", "Learning rate must be above 0.")
    assert fold.is_open()
    fold.button.click()
    assert fold.is_open() and not fold.body.isHidden(), "the fold closed over a field in error"
    rate.set_line("", "")
    assert fold.is_open()
    fold.button.click()
    assert not fold.is_open()
    rate.widget.setText("1,5")
    rate.widget.editingFinished.emit()
    assert fold.is_open(), "the field's own sentence did not open the fold"
    rate.widget.setText("1.5")
    fold.button.click()
    assert not fold.is_open()
    form.rows["num_runs"].set_line("err", "Batches is blank.")
    assert not fold.is_open()


def test_a_row_its_model_does_not_use_is_no_setting_of_the_fold():
    """A hidden row is not counted, not called not usual, not named in the line of values, and its error opens nothing. With
    every row hidden the fold takes no room."""
    form, rate, size, limit = _a_fold()
    rate.set_value(0.001)
    size.set_shown(False)
    assert form.fold.button.text() == "Rarely changed (2 settings, 1 not usual)"
    assert "Max rows" not in form.fold.summary.text()
    rate.set_shown(False)
    assert form.fold.button.text() == "Rarely changed (1 setting)"
    rate.set_line("err", "Learning rate must be above 0.")
    assert not form.fold.is_open() and not rate.in_error(), "a hidden row's error opened the fold"
    limit.set_shown(False)
    assert form.fold.isHidden()
    limit.set_shown(True)
    assert not form.fold.isHidden()


def test_a_value_that_is_only_shown_is_no_setting_of_the_fold():
    """It stands in the fold, and it is neither counted, nor called not usual, nor said in the line of values. A fold of
    nothing else still stands, for it holds something to read."""
    form, _rate, _size, _limit = _a_fold()
    form.add_text("chi_f0", "Drive amplitude in training: 0.15 (no unit).", "fixed by measurement", folded=True)
    form.rows["learning_rate"].set_value(0.0005)
    assert form.fold.button.text() == "Rarely changed (3 settings)"
    assert "Drive amplitude" not in form.fold.summary.text()
    alone = _the_form()
    alone.add_text("chi_f0", "Shown.", "fixed by measurement", folded=True)
    alone.add_text("chi_freq_bounds", "Shown too.", "fixed by measurement", folded=True)
    assert not alone.fold.isHidden() and alone.fold.button.text() == "Rarely changed (0 settings)"


@pytest.mark.parametrize("key", ["num_runs", "chi_f0"], ids=["a setting", "a value only shown"])
@pytest.mark.parametrize("name", [LONG_NAME, "x" * 60, "a.very.long.name.of.a.record.set.apart.by.full.stops.alone"],
                         ids=["underscores", "one word", "full stops"])
def test_a_long_origin_wraps_and_stays_on_the_page(name, key, tmp_path):
    """No record's name is too long for its row: the origin takes the lines it needs, a word with no place to break among
    them, at the window's least size, and its words are all still there."""
    from core.gui.widgets.form import _BREAK
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    row = page.form.rows[key]
    row.set_origin("from record " + name)
    _settle()
    port = page.scroller.viewport()
    assert _edge_of(row.origin, port).right() < port.width() and page.scroller.widget().width() <= port.width()
    assert row.origin.text().replace(_BREAK, "") == "from record " + name
    assert row.origin.height() > row.origin.fontMetrics().height() * 1.5, "it stood on one line, so this says nothing"
    row.set_origin("usual")
    assert row.origin.text() == "usual"


def test_a_row_off_the_page_reads_as_nothing_wrong():
    """A blank field a model does not use is no fault: a page need not remember to ask whether the row stands."""
    form = _the_form()
    row = form.add("chi_n_freqs", IntField(None))
    assert row.reading() == "Driven recordings per observation is blank."
    row.set_shown(False)
    assert row.reading() is None


def test_the_first_control_of_a_form_is_the_first_a_person_reads():
    """Whichever was added first: the main settings, the fold, the block for this run."""
    form = _the_form()
    form.add_name(build_part(NameAndNote, kind="prior", required=True))
    folded = IntField(1)
    form.add("n_cal", folded, usual=1, folded=True)
    field = IntField(1)
    form.add("num_runs", field)
    assert form.first_control() is field


def test_a_page_with_nothing_rarely_changed_shows_no_fold():
    form = _the_form()
    form.add("num_runs", IntField(1))
    assert form.fold.isHidden()


def test_the_form_says_every_edit_of_every_kind_of_field():
    """``edited`` hears a number, a list, a tick box, a ticked field and a text alike."""
    from PySide6.QtWidgets import QCheckBox
    form = _the_form()
    number, choice, tick, ticked, line = IntField(1), Choice(), QCheckBox(), TickedField(IntField(1), "Automatic", 0), QLineEdit()
    choice.addItems(["a", "b"])
    for key, widget in (("n_cal", number), ("chi_mode", choice), ("units", tick), ("run_size_cap", ticked), ("name", line)):
        form.add(key, widget, label=key)
    heard = []
    form.edited.connect(lambda: heard.append(1))
    for change in (lambda: number.setText("2"), lambda: choice.setCurrentIndex(1), lambda: tick.setChecked(True),
                   lambda: ticked.ticked_box.setChecked(True), lambda: line.setText("x")):
        heard.clear()
        change()
        assert heard, "a change that was not heard"


def test_two_columns_stand_side_by_side_on_a_wide_page_and_one_above_the_other_on_a_narrow_one():
    """``fit`` reaches the two columns, from the look's width; with no block for this run there is one column at any
    width."""
    form = _the_form()
    form.add("num_runs", IntField(1))
    form.add_name(build_part(NameAndNote, kind="prior", required=True))
    form.fit(design.two_columns_from())
    assert not form.columns.stacked()
    form.fit(design.two_columns_from() - 1)
    assert form.columns.stacked()
    alone = _the_form()
    assert alone.side.isHidden()


def test_the_folds_chevron_draws_at_every_scaling():
    """The look's own chevron, right while the fold is shut and down while it is open, sharp at the four scalings: read from
    the part of the button where the glyph stands and no letter does."""
    from tests._look_helpers import _BLANK, _screen_of, theme
    with theme(False), _the_look(False):
        form, _rate, _size, _limit = _a_fold()
        fold = form.fold
        fold.setVisible(True)
        fold.button.resize(fold.button.sizeHint())
        glyph = design.scaled(design.GAP["fold_icon"][1])

        def drawn(ratio):
            image = _screen_of(fold.button, ratio)
            return {(x, y) for y in range(int(fold.button.height() * ratio)) for x in range(int(glyph * ratio))
                    if image.pixelColor(x, y) != _BLANK}

        for ratio in (1, 1.25, 1.5, 2):
            shut = drawn(ratio)
            fold.button.click()
            opened = drawn(ratio)
            fold.button.click()
            assert shut and opened and shut != opened, ratio


# ── name and note ──
class _Store:
    """A store that counts the checks it is asked for and refuses the names it is told to."""

    def __init__(self, taken=()):
        self.taken, self.asked = taken, []

    def assert_name_free(self, kind, name):
        from core.artifacts import StoreError
        self.asked.append((kind, name))
        if name in self.taken:
            raise StoreError(f"a {kind} named {name!r} already exists; rename or delete it first", field="name")


def _names(*, taken=(), required=True):
    qt_app()
    store = _Store(taken)
    return build_part(NameAndNote, kind="prior", required=required, store=store), store


def test_the_name_is_read_after_the_typing_stops_and_again_at_the_press():
    """No keystroke reads the store; the timer, stopped until a keystroke starts it, reads once, however many keystrokes
    came; the press reads again. A name that passes says so beside the field. What is sent is stripped."""
    names, store = _names()
    assert not names.read_timer.isActive() and names.read_timer.isSingleShot() and names.read_timer.interval() == 300
    for typed in ("r", "re", "ret"):
        names.name.setText(typed)
    assert store.asked == [] and names.read_timer.isActive()
    names.read_timer.timeout.emit()
    names.read_timer.stop()
    assert store.asked == [("prior", "ret")]
    assert (names.rows["name"].line.label.text(), names.rows["name"].line.marker.kind) == (NAME_FREE_TEXT, "fine")
    assert NAME_FREE_TEXT == "Fine. This name is free."
    assert names.name_fact() is None
    names.name.setText("  retrain  ")
    names.check_now()
    assert store.asked[-1] == ("prior", "retrain") and not names.read_timer.isActive() and names.name_value() == "retrain"


def test_an_edit_takes_back_what_the_last_reading_said_and_a_blank_name_reads_nothing():
    names, store = _names()
    names.name.setText("a")
    names.check_now()
    names.name.setText("ab")
    assert names.rows["name"].line.isHidden(), "the sentence said of the last name stood under the new one"
    names.name.setText("")
    assert not names.read_timer.isActive() and len(store.asked) == 1


def test_a_taken_name_outlines_the_field_and_blocks_with_the_engines_sentence():
    names, _store = _names(taken=("retrain",))
    names.name.setText("retrain")
    names.check_now()
    sentence = "a prior named 'retrain' already exists; rename or delete it first"
    fact = names.name_fact()
    assert (fact.kind, fact.sentence, fact.blocks) == ("err", sentence, "the name cannot be used")
    assert names.rows["name"].line.label.text() == sentence and names.name.property("state") == "error"
    names.name.setText("free")
    names.check_now()
    assert names.name_fact() is None and names.name.property("state") == ""


def test_a_real_store_says_why_a_name_cannot_be_used(tmp_path):
    """A name that is no legal name and one shaped like an id, each in its own sentence, through the real call; a taken
    name has its own test."""
    from core.artifacts import ArtifactStore
    qt_app()
    names = build_part(NameAndNote, kind="prior", required=True, store=ArtifactStore(tmp_path))
    for typed, words in (("a b", "bad artifact name"), ("20261001T222935", "shaped like an artifact id")):
        names.name.setText(typed)
        names.check_now()
        assert words in names.name_fact().sentence and names.name_fact().blocks == BLOCKS


def test_a_store_that_cannot_be_read_says_nothing_of_the_name():
    """The stage reads the store again at its entry, so a failed reading here calls the name neither free nor taken."""
    names, store = _names()

    def unreadable(_kind, _name):
        raise OSError("the disk went away")

    store.assert_name_free = unreadable
    names.name.setText("a")
    names.check_now()
    assert names.name_fact() is None and names.rows["name"].line.isHidden()


def test_a_required_name_left_blank_blocks_with_its_own_words():
    names, store = _names()
    fact = names.name_fact()
    assert (fact.kind, fact.sentence, fact.blocks) == ("err", "Type a name.", "a name is needed")
    assert names.name.property("state") in (None, ""), "a blank name is no error to outline"
    assert store.asked == []
    optional, _store = _names(required=False)
    assert optional.name_fact() is None


def test_a_note_is_judged_at_every_keystroke_and_reads_no_disk():
    """Two hundred characters pass and 201 do not; a line edit takes a pasted line break out, so the rule's second cause
    is never met here. Each says so at once, and nothing is read."""
    names, store = _names()
    names.note.setText("x" * 200)
    assert names.note_fact() is None and names.note_value() == "x" * 200
    names.note.setText("x" * 201)
    fact = names.note_fact()
    assert fact.blocks == "the note cannot be used" and "at most 200 characters" in fact.sentence
    assert names.rows["note"].line.label.text() == fact.sentence and names.note.property("state") == "error"
    assert names.note_value() == ""
    names.note.setText("a\nb")
    assert (names.note_fact() is None) == ("\n" not in names.note.text())
    names.note.setText("  a note  ")
    assert names.note_fact() is None and names.note_value() == "a note" and names.note.property("state") == ""
    assert store.asked == [] and not names.read_timer.isActive()


def test_only_clear_empties_the_name_and_the_note():
    names, store = _names()
    names.name.setText("a")
    names.note.setText("b")
    names.set_hint("The earlier run was named retrain_3.")
    assert (names.hint.text(), names.hint.isHidden()) == ("The earlier run was named retrain_3.", False)
    names.check_now()
    assert (names.name_value(), names.note_value()) == ("a", "b")
    asked_before = len(store.asked)
    names.clear()
    assert (names.name.text(), names.note.text(), names.hint.text()) == ("", "", "") and names.hint.isHidden()
    assert not names.read_timer.isActive() and len(store.asked) == asked_before


# ── the foot ──
def test_the_keyboard_reaches_a_ticked_field_by_its_label_and_a_greyed_one_by_its_box(tmp_path):
    """A row says which of the two takes the keyboard: the field while it can, the box while it is greyed. No focus proxy
    stands on a ticked field: the toolkit would hand the keyboard to a greyed field, and a widget with one stops the Tab key
    at its own place in the order, which a page cannot move. The label's buddy is the field; greying the field that holds
    the keyboard hands it on."""
    from PySide6.QtWidgets import QApplication
    window, page, _clock = _a_page(tmp_path)
    form = page.form
    form.fold.set_open(True)
    _settle()
    row = form.rows["run_size_cap"]
    ticked = row.widget
    with in_front(window):
        assert ticked.is_ticked() and ticked.focusProxy() is None and row.label.buddy() is ticked.field
        assert row.focus_target() is ticked.ticked_box
        ticked.ticked_box.setChecked(False)
        assert row.focus_target() is ticked.field
        ticked.field.setFocus(Qt.OtherFocusReason)
        assert QApplication.focusWidget() is ticked.field
        ticked.ticked_box.setChecked(True)
        assert QApplication.focusWidget() is not ticked.field, "the keyboard stayed on a field that was greyed"


def test_the_name_is_not_read_while_a_run_is_live(tmp_path):
    """An edit waiting as a run starts is dropped, and the page reads it when the run ends: no reading of the store between
    the two, by the timer or by the page's own press."""
    window, page, _clock = _a_page(tmp_path)
    store = _Store(taken=("taken",))
    page.names._store = store
    page.names.name.setText("taken")
    assert page.names.read_timer.isActive()
    stand_in = StandIn()
    _start(page, stand_in)
    page.names.read_timer.timeout.emit()
    page.names.check_now()
    assert store.asked == [], "the name was read during a run"
    stand_in.release()
    turn_until(qt_app(), lambda: not controller().is_live())
    page.show_figures()
    assert store.asked == [("prior", "taken")] and page.names.name_fact().blocks == BLOCKS


def test_a_page_deleted_before_a_run_ends_reads_no_name_at_its_end(tmp_path):
    """A page deleted during a run takes its timers and its connections with it: the end reads nothing of it, and a reading
    waiting on its timer dies with it."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    store = _Store()
    page.names._store = store
    stand_in = StandIn()
    start_run(build_run_host(), stand_in)
    page.names.name.setText("another_name")
    timers = [page.names.read_timer, page.read_timer]
    _gone(window)
    assert not any(shiboken6.isValid(timer) for timer in timers)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert store.asked == []


def test_a_page_deleted_while_its_reading_waits_reads_nothing(tmp_path):
    """The timer goes with its page, so there is nothing to fire."""
    window, page, _clock = _a_page(tmp_path)
    page.read_soon()
    timer = page.read_timer
    assert timer.isActive()
    _gone(window)
    assert not shiboken6.isValid(timer)


def _facts(count, kind="info"):
    return tuple(Fact(kind, f"{TWO_LINES} ({at})") for at in range(count))


def test_a_facts_row_is_kept_across_a_reading():
    """A row whose fact did not change is not touched, one that changed is told its words and its marker in its place, and
    no row is built again; the last go, and one more is added, when the count changes."""
    qt_app()
    facts = build_part(Facts)
    first = (Fact("err", "One.", BLOCKS), Fact("info", "Two."), Fact("fine", "Three."))
    facts.set_facts(first)
    one, two, three = rows = facts.rows()
    said = two.sentence.plain_text()
    facts.set_facts(first)
    assert facts.rows() == rows
    facts.set_facts((Fact("warn", "One, later."), Fact("info", "Two."), Fact("fine", "Three.")))
    assert facts.rows() == [one, two, three]
    assert (one.kind, one.marker.kind, one.sentence.plain_text()) == ("warn", "warn", "Attention. One, later.")
    assert two.sentence.plain_text() == said == "Information. Two."
    facts.set_facts((Fact("info", "Two."), Fact("warn", "One, later."), Fact("fine", "Three."), Fact("fine", "Four.")))
    assert facts.rows()[:3] == [one, two, three] and len(facts.rows()) == 4
    assert (one.sentence.plain_text(), two.sentence.plain_text()) == ("Information. Two.", "Attention. One, later.")
    assert (one.kind, two.kind) == ("info", "warn"), "two rows swapped kept each other's markers"
    facts.set_facts((Fact("info", "Two."),))
    assert facts.rows() == [one]
    for row in facts.findChildren(MessageRow):
        QApplication.sendPostedEvents(row, QEvent.Type.DeferredDelete)
    assert facts.findChildren(MessageRow) == [one], "a row that went stands hidden in the region"
    facts.set_facts(())
    assert facts.rows() == [] and facts.isHidden()


def test_a_reading_keeps_the_place_a_person_scrolled_to(tmp_path):
    """Facts that scroll stay where they were scrolled when a reading changes one of them, moves two or adds one, and the
    keyboard stays on them; shrinking them takes the place with them."""
    from PySide6.QtWidgets import QApplication
    window, page, _clock = _a_page(tmp_path)
    page.said = _facts(9)
    page.redraw()
    _settle()
    scroller = page.foot.facts.scroller
    assert scroller.verticalScrollBar().maximum() > 0, "the facts do not scroll"
    with in_front(window):
        scroller.setFocus(Qt.TabFocusReason)
        bar = scroller.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)
        place = bar.value()
        rows = page.foot.facts.rows()
        page.said = (*_facts(8), Fact("warn", "Another."))
        page.redraw()
        _settle()
        assert bar.value() == place and QApplication.focusWidget() is scroller and page.foot.facts.rows() == rows
        page.said = (page.said[1], page.said[0], *page.said[2:], Fact("fine", "One more."))
        page.redraw()
        _settle()
        assert bar.value() == place and QApplication.focusWidget() is scroller
        page.said = ()
        page.redraw()
        _settle()
        assert page.foot.facts.isHidden() and QApplication.focusWidget() is not scroller


def test_the_main_button_is_enabled_exactly_when_nothing_blocks_and_no_run_is_live(tmp_path):
    """A blocking fact greys it, with the reason beside it; "Attention." does not; nothing to apply does; a run does, on
    this page and any other, and its end gives it back."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    foot = page.foot
    assert foot.main_button.isEnabled() and foot.reason.text() == ""
    page.said = (Fact("warn", "Mind this."), Fact("info", "And this."))
    page.redraw()
    assert foot.main_button.isEnabled()
    page.said = (Fact("info", "And this."), Fact("lock", "Needs a prior.", "needs a prior"))
    page.redraw()
    assert not foot.main_button.isEnabled() and foot.reason.text() == "Cannot start yet: needs a prior."
    assert [row.kind for row in foot.facts.rows()] == ["lock", "info"], "what blocks stands first"
    page.said = ()
    page.nothing = True
    page.redraw()
    assert not foot.main_button.isEnabled() and foot.reason.text() == "Nothing to apply: these choices are in use."
    page.nothing = False
    page.redraw()
    assert foot.main_button.isEnabled()
    stand_in = StandIn(word="Prior build")
    window.go_to("inference", "prior")
    start_run(window.part("inference").prior_panel, stand_in)
    assert not foot.main_button.isEnabled() and foot.reason.text() == LIVE_TEXT
    assert not foot.run_link.isHidden() and not foot.check_again.isEnabled()
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert foot.main_button.isEnabled() and foot.reason.text() == "" and foot.check_again.isEnabled()
    assert foot.run_link.isHidden()


def test_the_reason_is_the_buttons_accessible_description():
    qt_app()
    foot = build_part(Foot)
    foot.set_main_enabled(False, "Cannot start yet: the name cannot be used.")
    assert foot.main_button.accessibleDescription() == foot.reason.text() == "Cannot start yet: the name cannot be used."
    assert foot.reason.textFormat() == Qt.PlainText and foot.reason.font().family() != "Cambria"
    foot.set_main_enabled(True, "")
    assert foot.main_button.accessibleDescription() == "" and foot.main_button.isEnabled()


def test_the_main_buttons_words_carry_its_access_key():
    qt_app()
    foot = build_part(Foot)
    foot.set_main_words("Start training")
    assert foot.main_button.text() == "&Start training"
    foot.set_main_words("Load & go")
    assert foot.main_button.text() == "&Load && go"


def test_a_live_foot_stays_greyed_whatever_a_page_says_of_its_button():
    qt_app()
    foot = build_part(Foot)
    foot.set_live(True)
    foot.set_main_enabled(True, "")
    assert not foot.main_button.isEnabled() and not foot.check_again.isEnabled()
    foot.set_live(False)
    foot.set_main_enabled(True, "")
    assert foot.main_button.isEnabled() and foot.check_again.isEnabled()


def test_a_fact_row_never_greys_while_a_run_locks_the_page(tmp_path):
    """A run locks the form and the foot's two buttons, and nothing else: the facts, what is in use, the link beside the
    reason and the heading stay as they are, on a page the run was not started from, whose regions stay in sight."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    page.said = (Fact("err", "One.", BLOCKS), Fact("warn", "Two."))
    page.redraw()
    window.go_to("inference", "prior")
    stand_in = StandIn(word="Prior build")
    start_run(window.part("inference").prior_panel, stand_in)
    window.go_to("inference", "training")
    _settle()
    foot = page.foot
    assert not page.form.isEnabled() and not foot.main_button.isEnabled() and not foot.check_again.isEnabled()
    for part in (foot.facts, *foot.facts.rows(), foot.more, foot.run_link, foot.reading, foot.reason, page.in_use,
                 page.head, page.help_button):
        assert part.isEnabled(), type(part).__name__
    assert all(row.marker.isEnabled() for row in foot.facts.rows())
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert page.form.isEnabled() and foot.main_button.isEnabled() is False  # the facts still block


def test_the_low_window_shows_what_needs_attention_and_a_link_to_the_rest():
    """Told it is low, the facts keep what blocks or needs attention, and a link says how many there are; it opens them in
    the same region and shuts them again. With nothing left out there is no link. The ordinary window shows them all."""
    qt_app()
    foot = build_part(Foot)
    facts = (Fact("err", "a", BLOCKS), Fact("lock", "b", "c"), Fact("warn", "d"), Fact("info", "e"), Fact("fine", "f"))
    foot.set_facts(facts)
    rows = foot.facts.rows()
    assert foot.more.isHidden() and not any(row.isHidden() for row in rows)
    foot.set_small(True)
    assert [row.isHidden() for row in rows] == [False, False, False, True, True]
    assert (foot.more.isHidden(), foot.more.text()) == (False, "Show all 5")
    foot.more.click()
    assert not any(row.isHidden() for row in rows) and foot.more.text() == "Show fewer"
    foot.more.click()
    assert [row.isHidden() for row in rows] == [False, False, False, True, True] and foot.more.text() == "Show all 5"
    foot.set_small(False)
    assert foot.more.isHidden() and not any(row.isHidden() for row in rows)
    foot.set_small(True)
    foot.set_facts(facts[:3])
    assert foot.more.isHidden(), "a link with nothing left out"
    foot.set_facts(facts[3:])
    assert all(row.isHidden() for row in foot.facts.rows()) and foot.facts.isHidden()
    assert (foot.more.isHidden(), foot.more.text()) == (False, "Show all 2"), "all that was left out is the link's to show"


def test_show_all_stays_open_through_a_reading_a_move_and_a_run(tmp_path):
    """The window tells its pages the low form at every move to another screen, and a reading comes when it comes, a run's
    among them: none of these shuts what a person opened, on a page a run is not on. Only leaving the low form does,
    whether by a resize or by a move."""
    from types import SimpleNamespace
    app = qt_app()
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    page.said = (Fact("err", "a", BLOCKS), Fact("info", "b"), Fact("info", "c"))
    page.redraw()
    _settle()
    foot = page.foot
    assert window.shell._small and foot.more.text() == "Show all 3"
    foot.more.click()
    assert foot.more.text() == "Show fewer"

    def moved():
        window._screen_moved(SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 1000, 700)))
        _settle()
        assert foot.more.text() == "Show fewer" and not any(row.isHidden() for row in foot.facts.rows())

    moved()
    page.said = (Fact("err", "a", BLOCKS), Fact("info", "b, later"), Fact("info", "c"))
    page.redraw()
    assert foot.more.text() == "Show fewer"
    stand_in = StandIn(word="Reduction")
    start_run(window.part("reduction"), stand_in)
    moved()
    window.resize(901, 600)
    _settle()
    moved()
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    moved()
    window.resize(1000, 700)
    _settle()
    assert not window.shell._small and foot.more.isHidden()
    window.resize(900, 600)
    _settle()
    assert foot.more.text() == "Show all 3", "the low form came back open"


@pytest.mark.parametrize("factor", [1.0, 1.5])
@pytest.mark.parametrize("count", [0, 9, 12])
@pytest.mark.parametrize("size", [(900, 600), (1000, 699), (1000, 700), (1000, 820)], ids=str)
def test_the_foot_stays_over_nothing_at_the_smallest_window(factor, count, size, tmp_path):
    """At the window's least size and at the sizes on each side of the low form, with the fold open, a field in error and
    the facts of two lines each and a record's long name for an origin: the facts stand under their heading, no higher than
    the look's height, over nothing; the reason and the two buttons meet nothing and stand inside the window; the form
    keeps at least the room of a scrolling part, and is no wider than what shows it, which has no bar to reach the rest.
    Read in the window's own coordinates, and asked of the toolkit at each edge, for a part drawn over another or cut by a
    parent reads right by its edges alone. The facts open in the same region, and their growing moves nothing out of the
    window."""
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        wide, high = design.scaled(size[0]), design.scaled(size[1])
        shown_off_screen(window, wide, high)
        window.go_to("inference", "training")
        page.form.rows["learning_rate"].set_line("err", "Learning rate must be above 0.")
        page.names.name.setText("a name that is far too long " * 3)
        page.names.check_now()
        page.form.rows["num_runs"].set_origin("from record " + LONG_NAME)
        page.said = tuple(Fact("warn" if at % 2 else "info", f"{TWO_LINES} ({at})") for at in range(count))
        page.redraw()
        _settle()
        foot = page.foot
        assert page.form.fold.is_open() and (window.width(), window.height()) == (wide, high)

        def measured(why):
            facts = _edge_of(foot.facts, window)
            parts = {"reason": foot.reason, "check": foot.check_again, "main": foot.main_button}
            edges = {name: _edge_of(part, window) for name, part in parts.items()}
            if not foot.facts.isHidden():
                assert facts.height() <= design.scaled(FACTS_CAP), (why, facts.height())
                assert count < 9 or facts.height() == design.scaled(FACTS_CAP), (why, "the facts fall short of the cap")
                assert facts.top() >= _edge_of(foot.head, window).bottom(), (why, "the facts stand over their heading")
                for name, edge in edges.items():
                    assert not facts.intersects(edge), (why, name, facts, edge)
            for (one, first), (other, second) in itertools.combinations(edges.items(), 2):
                assert not first.intersects(second), (why, one, other)
            for name, part in parts.items():
                edge = edges[name]
                assert QRect(0, 0, wide, high).contains(edge), (why, name, edge)
                for corner in (edge.topLeft(), edge.bottomRight()):
                    found = window.childAt(corner)
                    assert found is part or (found is not None and part.isAncestorOf(found)), (why, name, corner, found)
            assert page.scroller.height() >= 56, (why, page.scroller.height())
            assert page.scroller.geometry().bottom() < page.scroller.parentWidget().height(), why
            assert page.scroller.widget().width() <= page.scroller.viewport().width(), (why, "the form runs off the page")
            if count:
                last = foot.facts.rows()[0] if foot.facts.isHidden() else [
                    row for row in foot.facts.rows() if not row.isHidden()][-1]
                assert last.width() > 0, why

        measured((factor, count, size, "as it stands"))
        if not foot.more.isHidden():
            foot.more.click()
            _settle()
            measured((factor, count, size, "all shown"))
            foot.more.click()
            _settle()


@pytest.mark.parametrize("factor", [1.5, 2.0, 2.25])
def test_large_text_on_a_small_screen_keeps_every_fact_in_reach(factor, tmp_path):
    """A screen of 1366 x 728 cannot give the look's height at large text. The facts keep to what there is, so that the last
    of them can be scrolled to, the form keeps the room of a scrolling part, and the button stays in the window."""
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        room = (1366, 728)
        design.set_room(room)
        _opening, least = design.window_sizes(room)
        shown_off_screen(window, *least)
        window.go_to("inference", "training")
        page.said = _facts(9, "warn")
        page.redraw()
        _settle()
        foot, facts = page.foot, page.foot.facts
        assert facts.scroller.height() == facts.height(), "the region cuts its own scroller"
        assert page.scroller.height() >= 56 and QRect(0, 0, *least).contains(_edge_of(foot.main_button, window))
        bar = facts.scroller.verticalScrollBar()
        bar.setValue(bar.maximum())
        _settle()
        last = _edge_of(facts.rows()[-1], facts.scroller.viewport())
        assert last.bottom() < facts.scroller.viewport().height(), "the last fact is out of reach"


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_a_long_reason_wraps_beside_the_buttons_and_pushes_nothing_out(factor, tmp_path):
    """The longest reason there is, and the longest main button, at the window's least size: the reason takes the lines it
    needs, its words all in its own rectangle, and the row stays in the window."""
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        wide, high = design.scaled(900), design.scaled(600)
        shown_off_screen(window, wide, high)
        window.go_to("inference", "training")
        page.said = (Fact("err", "a", "a saved simulation set would be continued"), Fact("err", "b", "x"), Fact("err", "c", "y"))
        page.redraw()
        _settle()
        foot = page.foot
        reason, main = _edge_of(foot.reason, window), _edge_of(foot.main_button, window)
        assert foot.reason.text() == "Cannot start yet: a saved simulation set would be continued, and 2 more above."
        assert foot.reason.heightForWidth(reason.width()) <= reason.height(), "the reason was cut"
        assert foot.reason.heightForWidth(reason.width()) > foot.reason.fontMetrics().height() * 1.5, (
            "the reason stood on one line, so this says nothing")
        assert not reason.intersects(main) and QRect(0, 0, wide, high).contains(main)
        assert foot.main_button.width() >= foot.main_button.sizeHint().width(), "the reason squeezed the main button"


def test_a_facts_region_that_grows_after_a_reading_keeps_clear_of_the_buttons(tmp_path):
    """Rows that come after a reading take their room at once, with no resize to tell them; and one less gives it back."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    foot = page.foot
    _settle()
    room = page.scroller.height()
    page.said = _facts(2, "warn")
    page.redraw()
    _settle()
    two = foot.facts.height()
    assert 0 < two < design.scaled(FACTS_CAP) and page.scroller.height() == room - two, (two, room)
    assert not _edge_of(foot.facts, window).intersects(_edge_of(foot.main_button, window))
    page.said = _facts(9, "warn")
    page.redraw()
    _settle()
    assert foot.facts.height() == design.scaled(FACTS_CAP)
    assert not _edge_of(foot.facts, window).intersects(_edge_of(foot.main_button, window))
    page.said = _facts(1, "warn")
    page.redraw()
    _settle()
    assert foot.facts.height() < two
    page.said = ()
    page.redraw()
    _settle()
    assert page.scroller.height() == room


def test_facts_that_come_take_the_height_of_the_width_they_stand_at(tmp_path):
    """A region with nothing in it stands hidden, and keeps no width: the first rows are measured again once it has one, so
    that two lines at a wide window take the height of two lines."""
    window, page, _clock = _a_page(tmp_path, size=(1900, 820))
    foot = page.foot
    page.said = _facts(1, "info")
    page.redraw()
    _settle()
    row = foot.facts.rows()[0]
    assert foot.facts.scroller.verticalScrollBar().maximum() == 0
    assert foot.facts.height() == row.heightForWidth(row.width()) + 4, (foot.facts.height(), row.width())


def test_rows_of_one_line_take_what_they_need_and_no_more(tmp_path):
    """The height is what the rows need at this width, not the cap, and it follows the window's width."""
    window, page, _clock = _a_page(tmp_path, size=(1300, 820))
    foot = page.foot
    page.said = (Fact("info", "One line."), Fact("info", "One line too."))
    page.redraw()
    _settle()
    assert foot.facts.height() < design.scaled(FACTS_CAP) / 2
    assert foot.facts.scroller.verticalScrollBar().maximum() == 0, "rows that fit were left to scroll"
    page.said = _facts(2, "info")
    page.redraw()
    _settle()
    wide = foot.facts.height()
    window.resize(900, 820)
    _settle()
    assert foot.facts.height() > wide, "the height did not follow the width"


# ── "In use" ──
def test_in_use_says_what_the_step_holds_or_that_it_holds_nothing():
    qt_app()
    in_use = build_part(InUse)
    assert in_use.line_text() == "In use: nothing yet." and in_use.sentence.isHidden()
    in_use.set_held("", "Choose the model and the kind of observation, then apply them.")
    assert in_use.line_text() == "In use: nothing yet."
    assert in_use.sentence.plain_text() == "Choose the model and the kind of observation, then apply them."
    in_use.set_held("retrain_<1>", "Built on master.txt.")
    assert in_use.line_text() == "In use: retrain_<1>" and not in_use.sentence.isHidden()
    assert "retrain_&lt;1&gt;" in in_use.line.text(), "a name was read as markup"
    in_use.set_held("x", "")
    assert in_use.sentence.isHidden()


def test_in_use_counts_the_choices_not_applied():
    qt_app()
    in_use = build_part(InUse)
    assert in_use.pending.isHidden()
    in_use.set_pending(1)
    assert in_use.pending.sentence.plain_text() == "Attention. 1 choice below is changed and not applied."
    in_use.set_pending(2)
    assert in_use.pending.sentence.plain_text() == "Attention. 2 choices below are changed and not applied."
    in_use.set_pending(0)
    assert in_use.pending.isHidden()


def test_a_row_that_picks_what_the_step_holds_says_when_it_differs_and_the_rail_hears(tmp_path):
    """A page names its selection rows and the step page does the rest: under a row that shows something other than what
    is held, "Changed, not applied" and what is in use, counted in "In use", and the rail told once each way; a line of
    the page's own that stops the start stands in its place, and the line goes when the choice is put back."""
    from core.gui.step_facts import SelectionRow
    window, page, _clock = _a_page(tmp_path)
    screen, held = window.part("inference"), [False]
    page.selection_rows = lambda: (SelectionRow("chi_mode", lambda: held[0], page.kind_choice.currentData,
                                                lambda value: "Frequency response" if value else "Set by the bounds file"),)
    heard = []
    screen.states_changed.connect(lambda states: heard.append(states[2].rail_words))
    row = page.form.rows["chi_mode"]
    page.kind_choice.setCurrentIndex(1)
    assert (row.line.kind, row.line.words) == ("warn", "Changed, not applied. In use: Set by the bounds file.")
    assert page.in_use.pending.sentence.plain_text() == "Attention. 1 choice below is changed and not applied."
    assert heard == ["Changed, not applied"] and screen.pending() == {"training"}
    page.redraw()
    page.refresh_local_gates()
    assert heard == ["Changed, not applied"], "a drawing told the rail again"
    page.kind_choice.setCurrentIndex(0)
    assert not row.line.says() and page.in_use.pending.isHidden() and screen.pending() == frozenset()
    assert heard == ["Changed, not applied", "Needs a prior"], heard
    page.facts_from_form = lambda: (row.set_line("err", "Not that one."), ())[1]
    page.kind_choice.setCurrentIndex(1)
    assert (row.line.kind, row.line.words) == ("err", "Not that one.")
    assert not page.in_use.pending.isHidden(), "a change under an error was not counted"


def test_a_pages_own_line_stands_under_what_is_in_use_and_goes_when_its_next_run_starts(tmp_path):
    """A line the page says is a marked line that blocks nothing, even of a level that would be an error, and a run started
    here takes it away."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    page.own_line("It was saved without its name: 'retrain' was taken while the run was going.", "warning")
    said = page.in_use.said
    assert (said.isHidden(), said.kind) == (False, "warn")
    assert said.sentence.plain_text().startswith("Attention. It was saved without its name")
    page.own_line("Something went wrong.", "error")
    assert said.kind == "warn", "a line of the page's own would be an error that blocks nothing"
    page.own_line("Read.")
    assert said.kind == "info"
    stand_in = StandIn()
    _start(page, stand_in)
    assert said.isHidden()
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


# ── when a page reads ──
@pytest.mark.parametrize("page_class", list(PAGES.values()), ids=list(PAGES))
def test_a_page_is_built_with_no_reading_and_no_running_timer(page_class, tmp_path, monkeypatch):
    """A constructor reads none of what a run may not have read and starts nothing: no card, no program, no reading of the
    disk's facts, no timer running. A window that is never shown reads nothing either."""
    from PySide6.QtCore import QTimer
    qt_app()
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    window = build_window()
    window.part("inference")
    calls = readings_of_a_run(monkeypatch)
    page = build_step_page(page_class, window=window)
    _settle()                                # what a constructor left for the loop's next turn counts too
    assert page.stale, "a page starts with its reading owed"
    assert getattr(page, "readings", 0) == 0
    assert not [timer for timer in page.findChildren(QTimer) if timer.isActive()], "a timer runs on a page just built"
    assert calls == [], calls


def test_a_page_reads_once_as_it_is_first_shown_and_not_again_at_each_show(tmp_path):
    """``showEvent`` reads while a reading is owed, once; a page shown again with nothing owed reads nothing."""
    window, page, _clock = _a_page(tmp_path, shown=False)
    page.readings = 0
    shown_off_screen(window, 1300, 820)
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 1 and not page.stale
    window.go_to("inference", "prior")
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 1


def test_a_reading_waits_for_the_typing_to_stop(tmp_path):
    """Ten edits read nothing; the timer, once it fires, reads once. A page not in sight starts no timer: it reads at its
    next show. An edit that is drawn is drawn at once: only the reading waits."""
    window, page, _clock = _a_page(tmp_path)
    field = page.form.rows["num_runs"].widget
    assert page.read_timer.isSingleShot() and page.read_timer.interval() == 300
    for digit in range(10):
        field.setText(f"12{digit}")
        assert page.readings == 0
    assert page.read_timer.isActive()
    page.read_timer.timeout.emit()
    page.read_timer.stop()
    assert page.readings == 1
    window.go_to("inference", "prior")
    _settle()
    field.setText("7")
    assert not page.read_timer.isActive() and page.stale
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 2


def test_a_fact_of_the_form_follows_every_edit_at_once(tmp_path):
    """What the form and the session say needs no reading: it is drawn at the keystroke, and only the disk's facts wait."""
    window, page, _clock = _a_page(tmp_path)
    field = page.form.rows["num_runs"].widget
    page.facts_from_form = lambda: (Fact("err", "Too many.", "too many"),) if field.value_or_none() == 7 else ()
    field.setText("7")
    assert page.foot.reason.text() == "Cannot start yet: too many." and page.readings == 0
    field.setText("6")
    assert page.foot.reason.text() == "" and page.readings == 0


def test_a_timer_that_fires_on_a_page_left_reads_at_its_next_show(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    page.form.rows["num_runs"].widget.setText("7")
    window.go_to("inference", "prior")
    _settle()
    page.read_timer.timeout.emit()
    assert page.readings == 0 and page.stale
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 1


def test_check_again_reads(tmp_path):
    window, page, clock = _a_page(tmp_path)
    clock.advance(600)
    page.foot.check_again.click()
    assert page.readings == 1 and page.foot.reading.text() == "The saved records were read at 12:14."


def test_a_reading_says_when_on_the_controllers_clock_and_a_run_changes_the_line(tmp_path):
    """The line follows the run's status, with no reading: a run's start says the reading stands from before it, and its end
    reads again."""
    app = qt_app()
    window, page, clock = _a_page(tmp_path, clock=Clock(at(8, 9, 7)))
    assert page.foot.reading.text() == "The saved records were read at 09:07."
    stand_in = StandIn()
    window.go_to("inference", "prior")
    start_run(window.part("inference").prior_panel, stand_in)
    window.go_to("inference", "training")
    _settle()
    assert page.foot.reading.text() == "A run is live. Read at 09:07; read again when it ends."
    clock.advance(120)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert page.readings == 1 and page.foot.reading.text() == "The saved records were read at 09:09."


@pytest.mark.parametrize("page_class", list(PAGES.values()), ids=list(PAGES))
@pytest.mark.parametrize("left_for", ["the end", "the first opening"])
def test_no_reading_happens_between_a_runs_start_and_its_end(left_for, page_class, tmp_path, monkeypatch):
    """Nothing a page reads is read between a run's start and its end: a page opened during the run, an edit waiting as the
    run starts, the timer firing, "Check again" and the press each read nothing, and the run's end reads once on a page in
    sight, or at its next show on one that is not. Parameter inference's old pages have their own count."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path, shown=left_for == "the end", cls=page_class)
    if left_for == "the first opening":
        shown_off_screen(window, 1300, 820)
    runner = window.part("reduction")
    counted = {"readings": 0, "pressed": 0}
    for name, kept in (("readings", page.read_disk), ("pressed", page.main_pressed)):
        monkeypatch.setattr(page, "read_disk" if name == "readings" else "main_pressed",
                            lambda name=name, kept=kept: (counted.__setitem__(name, counted[name] + 1), kept())[1])
    readings = 0
    page.form.rows[EDITED[page_class]].widget.setText("7")
    stand_in = StandIn(word="Reduction")
    calls = readings_of_a_run(monkeypatch)
    start_run(runner, stand_in)
    if left_for == "the first opening":
        page.stale = True
        window.go_to("inference", "training")
        _settle()
        assert page.foot.reading.text() == NOT_READ_YET_TEXT
    assert counted["readings"] == readings
    page.read_timer.timeout.emit()
    page.read_soon()
    page.foot.check_again.click()
    page.read_facts()
    page.foot.main_button.click()
    page._main_clicked()
    page.stale_store()
    page.refresh_local_gates()
    page.redraw()
    assert (counted["readings"], counted["pressed"]) == (readings, 0), "a page read during a run"
    assert calls == []
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert counted["readings"] == readings + 1, "the run's end did not read once"
    assert not page.read_timer.isActive(), "a reading still waits after the run's end has read"


def test_a_runs_end_reads_a_page_not_in_sight_at_its_next_show(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    window.go_to("inference", "prior")
    stand_in = StandIn(word="Prior build")
    start_run(window.part("inference").prior_panel, stand_in)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert page.readings == 0 and page.stale
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 1


def test_a_store_change_reads_a_shown_page_now_and_a_hidden_one_at_its_show(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    inference = window.part("inference")
    inference.stale_store()
    assert page.readings == 1
    window.go_to("inference", "prior")
    _settle()
    inference.stale_store()
    assert page.readings == 1 and page.stale
    window.go_to("inference", "training")
    _settle()
    assert page.readings == 2


def test_a_store_change_during_a_run_reads_at_its_end(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    runner = window.part("reduction")
    stand_in = StandIn(word="Reduction")
    start_run(runner, stand_in)
    window.part("inference").stale_store()
    assert page.readings == 0
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert page.readings == 1


def test_a_start_taken_back_reads_as_a_runs_end(tmp_path):
    """A start that raises after its page was handed the run sends no end of a run: the page is given back, the reading line
    leaves "A run is live." at once, and the reading comes once, as the timer fires."""
    window, page, _clock = _a_page(tmp_path)
    with pytest.raises(TypeError):
        _start(page, StandIn(), watch_dir=5)
    assert not controller().is_live() and page.results.currentWidget() is page.figures
    assert page.form.isEnabled() and page.foot.check_again.isEnabled() and page.foot.reason.text() != LIVE_TEXT
    assert page.foot.reading.text().startswith("The saved records were read"), page.foot.reading.text()
    assert page.read_timer.isActive() and page.readings == 0
    page.read_timer.timeout.emit()
    page.read_timer.stop()
    assert page.readings == 1


def test_a_start_taken_back_at_its_first_drawing_leaves_no_word_of_a_run(tmp_path, monkeypatch):
    """A first drawing that raises comes before the status is said, so the page never heard a run was live, and it was
    given back while the run was still called live: the status that follows draws it again."""
    from core.gui.run.controller import RunController
    window, page, _clock = _a_page(tmp_path)

    def broken(_self):
        raise TypeError("the first drawing failed")

    monkeypatch.setattr(RunController, "_draw", broken)
    with pytest.raises(TypeError):
        _start(page, StandIn())
    assert not controller().is_live() and page.foot.reason.text() == "" and page.foot.main_button.isEnabled()
    assert page.foot.reading.text().startswith("The saved records were read"), page.foot.reading.text()
    assert not page.read_timer.isActive() and page.readings == 0, "no run ran, so nothing is owed"


def test_a_status_that_changes_nothing_reads_nothing(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    held = controller()
    held.status_changed.emit(held.status)
    assert page.readings == 0 and not page.read_timer.isActive()


def test_a_press_is_the_reading_that_counts(tmp_path):
    """The press reads, the name among it, and goes on; when that reading finds what blocks, the button greys with its reason
    and the page's own press is not made."""
    window, page, _clock = _a_page(tmp_path)
    store = _Store(taken=("taken",))
    page.names._store = store
    page.names.name.setText("free")
    page.names.check_now()
    store.asked.clear()
    page.foot.main_button.click()
    assert (page.pressed, page.readings, store.asked) == (1, 1, [("prior", "free")])
    store.taken = ("free",)
    page.foot.main_button.click()
    assert page.pressed == 1 and not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: the name cannot be used."


def test_a_second_press_starts_no_second_run(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    runs = []
    page.main_pressed = lambda: runs.append(_start(page, stand_in))
    stand_in = StandIn()
    page.foot.main_button.click()
    page.foot.main_button.click()
    page._main_clicked()
    assert len(runs) == 1 and len(stand_in.calls) <= 1
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    page.show_figures()


def test_a_page_deleted_with_its_window_leaves_no_connection_on_the_controller(tmp_path):
    """The two signals of the process's controller and the application's change of focus hear a page through methods of its
    own, so that a page deleted leaves nothing behind: the count of what hears them is what it was, and a run after that
    reads nothing of it."""
    app = qt_app()
    held = controller()

    def hearing():
        return (held.receivers("2status_changed(PyObject)"), held.receivers("2run_ended(PyObject)"),
                app.receivers("2focusChanged(QWidget*,QWidget*)"))

    before = hearing()
    window, page, _clock = _a_page(tmp_path)
    during = hearing()
    assert all(now > then for now, then in zip(during, before)), (before, during)
    _gone(window)
    assert hearing() == before
    held.status_changed.emit(held.status)
    held.run_ended.emit(held.status)


# ── the page, and a run on it ──
def test_a_page_with_nothing_saved_to_read_says_no_reading_and_offers_no_second_one(tmp_path):
    """Asked at every drawing, so that a page whose answer changes shows and hides the two at its next one, and a hidden
    "Check again" is no stop of the walk."""
    from tests._fixtures import _the_stand_in_page

    class Quiet(_the_stand_in_page()):
        has_a_file = False

        @property
        def reads_the_disk(self):
            return self.has_a_file

    fonts.load_app_font(qt_app())
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    window = build_window()
    design.set_room((3400, 1380))
    window.setMinimumSize(1, 1)
    page = build_step_page(Quiet, window=window)
    shown_off_screen(window, 1300, 820)
    window.go_to("inference", "training")
    _settle()
    foot = page.foot
    assert foot.reading.isHidden() and foot.check_again.isHidden() and not foot.main_button.isHidden()
    page.names.name.setText("retrain_3")
    page.names.check_now()
    with in_front(window):
        page.names.note.setFocus(Qt.TabFocusReason)
        assert _walked(window, 2) == [page.names.note, foot.main_button], "a hidden button was a stop"
    page.has_a_file = True
    page.redraw()
    assert not foot.reading.isHidden() and not foot.check_again.isHidden()
    page.has_a_file = False
    page.redraw()
    assert foot.reading.isHidden() and foot.check_again.isHidden()


def test_a_page_reads_as_the_default(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    assert page.reads_the_disk is True and not page.foot.check_again.isHidden() and not page.foot.reading.isHidden()


def test_the_page_stands_heading_in_use_form_and_foot_in_that_order(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    tops = [_edge_of(part, window).top() for part in (page.head, page.in_use, page.form, page.foot)]
    assert tops == sorted(tops) and len(set(tops)) == 4, tops
    assert page.scroller.isAncestorOf(page.in_use) and page.scroller.isAncestorOf(page.form)
    assert not page.scroller.isAncestorOf(page.foot), "the foot scrolls"
    assert page.controls is page.form and page.results.currentWidget() is page.figures


def test_a_rebuilt_page_stands_at_homes_distance_from_the_rail(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    rail = window.shell.rail.width()
    assert _edge_of(page.head, window).left() - rail == design.SIZE["page_pad"][0]


def test_a_middle_begins_with_no_control(tmp_path):
    """The first things in the scrolling body are the plan in use and what is in use, neither of which takes a key: a
    control there has no room for the top of its ring."""
    window, page, _clock = _a_page(tmp_path)
    layout = page.scroller.body.layout()
    assert [layout.itemAt(at).widget() for at in (0, 1)] == [page.plan_line, page.in_use]
    assert page.plan_line.focusPolicy() == page.in_use.focusPolicy() == Qt.NoFocus


def test_the_run_page_takes_the_place_of_the_form_and_gives_it_back(tmp_path):
    """A run started here takes the place of everything under the heading, the heading staying, and gives it back at a
    finish, with "How the run went" above the figures; that link brings it again, and "Back to the form" takes it."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    assert page.results.currentWidget() is page.run_view and page.head.isVisible() and not page.form.isVisible()
    assert not page.form.isEnabled() and not page.foot.main_button.isEnabled()
    assert page.is_running()
    stand_in.figure("Training loss")
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.results.currentWidget() is page.figures and page.form.isEnabled()
    assert page.run_view.how_link.isVisibleTo(page.figures)
    assert page.scroller.isAncestorOf(page.figure_stack), "the figures did not come back to their place"
    page.run_view.how_link.click()
    assert page.results.currentWidget() is page.run_view
    page.run_view.back_button.click()
    assert page.results.currentWidget() is page.figures


@pytest.mark.parametrize("width", (1300, 1000, 900))
def test_the_run_page_stands_at_the_edges_of_the_heading_and_the_form(tmp_path, width):
    """The run's page takes the place of the form, so it starts where the heading starts and ends where the heading's own
    buttons end; a run that moved them by the room for a focus ring would shift the page under the owner's eye."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path, size=(width, 820))
    form = _edge_of(page.foot.main_button, window).right()
    stand_in = StandIn()
    _start(page, stand_in)
    _settle()
    title, stop = _edge_of(page.run_view.title, window), _edge_of(page.run_view.stop_button, window)
    assert title.left() == _edge_of(page.head, window).left()
    assert stop.right() == _edge_of(page.help_button.parentWidget(), window).right() == form
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


def test_the_figures_stand_in_their_place_and_hide_while_they_hold_nothing(tmp_path):
    """They stand between the link and the form, come into sight with their first figure and go with their last, and through
    a run on the run's page they are in sight there whether they held anything or not."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stack = page.figure_stack
    assert stack.isHidden()
    stack.add_figure("A figure", b"")
    assert not stack.isHidden()
    stack.clear_all()
    assert stack.isHidden()
    stand_in = StandIn()
    _start(page, stand_in, provide_fig_sink=True)
    assert page.run_view.isAncestorOf(stack) and not stack.isHidden()
    stand_in.figure("A figure")
    turn_until(app, lambda: stack.count() == 1)
    _settle()
    assert stack.isVisible() and page.run_view.figures.isVisible(), "the run's figure was not in sight on its page"
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    box = page.scroller.body.layout()
    assert box.indexOf(stack) == box.indexOf(page.run_view.how_link) + 1 and stack.isVisible()


@pytest.mark.parametrize("end", ["returns its record", "is refused", "fails", "is stopped"])
def test_a_run_empties_the_figures_of_the_page_it_starts_on(end, tmp_path):
    """One record's figures at a time: a run that takes the column empties the viewer as it starts, and once it has
    finished "How the run went" opens its page on its figures alone. A load, which never takes the page, keeps every
    figure while it runs and leaves that run's page as it was. One that returns its record closes the figures that are
    not its own, and "How the run went" goes, for the page then shows that record alone and no run's page is the load's;
    one that returns none, refused, failed or stopped, closes what it added and leaves the figures, the link and the
    run's page as they stood. A start on another page empties nothing here."""
    from core.gui.run.model import RunSpec
    from core.refusals import Refusal
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    page.figure_stack.add_figure("Held", b"")
    loading = RunSpec("Prior load", ("inference", "training"), keeps_files=False, loads_only=True, stop_asks=False)
    stack, link = page.figure_stack, page.run_view.how_link

    def titles(shown=False):
        return [stack.tabText(at) for at in range(stack.count()) if not shown or stack.isTabVisible(at)]

    build = StandIn()
    _start(page, build, provide_fig_sink=True)
    assert titles() == []
    build.figure("Built")
    turn_until(app, lambda: stack.count() == 1)
    build.release()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert titles() == ["Built"] and link.isVisibleTo(page.figures)
    load = StandIn(word="Prior load")
    _start(page, load, spec=loading, provide_fig_sink=True)
    if end != "is refused":
        load.figure("Prior (loaded)")
        turn_until(app, lambda: stack.count() == 2)
    if end == "returns its record":
        load.release()
    elif end == "is refused":
        load.fail(Refusal("The prior is for another model.", field="prior"))
    elif end == "fails":
        load.fail(RuntimeError("the record cannot be read"))
    else:
        controller().request_stop()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.results.currentWidget() is page.figures
    if end == "returns its record":
        assert titles() == ["Prior (loaded)"] and not link.isVisibleTo(page.figures), (titles(), end)
    else:
        assert titles() == ["Built"] and link.isVisibleTo(page.figures), (titles(), end)
        link.click()
        assert page.results.currentWidget() is page.run_view and titles(shown=True) == ["Built"], end
        page.run_view.back_button.click()
        _settle()
        assert page.results.currentWidget() is page.figures and link.isVisibleTo(page.figures)
    other = StandIn(word="Reduction")
    start_run(window.part("reduction"), other)
    assert page.figure_stack.count() == 1
    other.release()
    turn_until(app, lambda: not controller().is_live())
    stand_in = StandIn()
    _start(page, stand_in)
    assert page.figure_stack.count() == 0
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


def test_a_failure_no_runs_page_shows_is_told_in_the_shared_question(tmp_path):
    """A load's failure, which no run's page shows, is told in the shared question, the error's type and its trace
    behind its button, a stop asked for meanwhile said beneath, and it stands after as the page's own line; a refusal is
    told as every refusal is."""
    from core.gui.run.retry import NOTED_TEXT
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    page._on_error(OSError("the file is locked"), "Traceback (most recent call last):\nOSError: the file is locked\n")
    box = asked()
    assert (box.kind, box.title, box.marker, box.sentence, box.answers, box.beneath) == (
        "error", "Error", "err", "the file is locked", ("OK",), "")
    assert box.detail == ("OSError: the file is locked\n\nTraceback (most recent call last):\n"
                          "OSError: the file is locked")
    said = page.in_use.said
    assert not said.isHidden() and said.sentence.plain_text() == "Attention. the file is locked"
    page._on_error(RuntimeError(), "", cancel_noted=True)
    box = asked()
    assert (box.sentence, box.beneath, box.detail) == ("RuntimeError", NOTED_TEXT, "RuntimeError")
    page._on_error(Refusal("The prior is for another model.", field="prior"), "")
    assert asked().kind == "refusal"


@pytest.mark.parametrize("how", ["a stop", "a failure"])
def test_a_stop_and_a_failure_keep_the_run_page_until_back_to_the_form(how, tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    if how == "a stop":
        controller().request_stop()
    else:
        stand_in.fail(RuntimeError("the stand-in failed"))
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.results.currentWidget() is page.run_view and page.form.isEnabled()
    page.run_view.back_button.click()
    _settle()
    assert page.results.currentWidget() is page.figures and page.run_view.how_link.isVisibleTo(page.figures)


@pytest.mark.parametrize("width", (1300, 900))
def test_a_past_view_stands_at_the_edges_of_the_heading_and_the_form(tmp_path, width):
    """A past run takes the same place as the run's page, so it stands at the same edges: both are made in one place."""
    from core.gui.run import history
    window, page, _clock = _a_page(tmp_path, size=(width, 820))
    form = _edge_of(page.foot.main_button, window).right()
    run = "20261001T090000-training"
    history.write(tmp_path / "runs", {"id": run, "word": "Training", "started": at(1, 9, 0), "ended": at(1, 9, 30),
                                      "outcome": "finished", "place": ["inference", "training"]})
    assert controller().show_past(run, page)
    _settle()
    title, back = _edge_of(page.past_view.title, window), _edge_of(page.past_view.back_button, window)
    assert title.left() == _edge_of(page.head, window).left() and back.right() == form


def test_a_past_view_stands_in_the_stack_and_gives_the_column_back(tmp_path):
    from core.gui.run import history
    window, page, _clock = _a_page(tmp_path)
    held = controller()
    run = "20261001T090000-training"
    history.write(tmp_path / "runs", {"id": run, "word": "Training", "started": at(1, 9, 0), "ended": at(1, 9, 30),
                                      "outcome": "finished", "place": ["inference", "training"]})
    assert held.show_past(run, page)
    assert page.results.currentWidget() is page.past_view and page.past_view.isVisible()
    page.past_view.back_button.click()
    assert page.results.currentWidget() is page.figures


def test_a_page_is_told_the_low_window_through_its_screen_and_its_foot_hears(tmp_path):
    """The heading goes and the foot keeps what needs attention below the look's height; the form's two columns follow the
    width the scroller tells the body."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    assert page.page.is_small() and page.foot.facts._small
    assert page.head.sentence.isHidden() and page.form.columns.stacked()
    window.resize(1300, 800)
    _settle()
    assert not page.page.is_small() and not page.foot.facts._small and not page.head.sentence.isHidden()
    assert not page.form.columns.stacked(), "the body did not hand the form its width"


def test_a_page_hands_the_keyboard_its_first_setting_and_none_while_it_is_locked(tmp_path):
    """The first setting, never a button; a first row that cannot take the keyboard is passed over. While a run locks the
    form none: a step first opened then leaves the keyboard on its row of the rail."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    inference = window.part("inference")
    assert page.first_focus() is page.form.rows["num_runs"].widget
    assert inference.first_focus() is page.form.rows["num_runs"].widget
    page.form.rows["num_runs"].set_shown(False)
    assert page.first_focus() is page.kind_choice
    page.form.rows["num_runs"].set_shown(True)
    page.form.rows["num_runs"].widget.setEnabled(False)
    assert page.first_focus() is page.kind_choice
    page.form.rows["num_runs"].widget.setEnabled(True)
    stand_in = StandIn(word="Reduction")
    start_run(window.part("reduction"), stand_in)
    assert page.first_focus() is None
    window.go_to("inference", "training")
    _settle()
    with in_front(window):
        window.go_to("inference", "training")
        from PySide6.QtWidgets import QApplication
        assert QApplication.focusWidget() is window.shell.rail.row("training")
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert page.first_focus() is page.form.rows["num_runs"].widget


def test_a_page_asks_the_window_for_help_and_for_the_next_step_through_its_screen(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    assert page.next_button.isHidden()
    page.set_next("Go to Checks", "checks")
    assert page.next_button.text() == "Go to Checks" and not page.next_button.isHidden()
    page.next_button.click()
    assert window.current_place() == ("inference", "checks")
    window.go_to("inference", "training")
    page.set_next("", None)
    assert page.next_button.isHidden()
    page.help_button.click()
    assert window.current_place() == ("help", None)


def test_go_to_the_run_goes_to_the_run(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn(word="Reduction")
    start_run(window.part("reduction"), stand_in)
    page.foot.run_link.click()
    assert window.current_place() == ("reduction", None)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


def test_enter_presses_each_new_button_with_the_ring_and_nothing_in_a_field(tmp_path, monkeypatch):
    """The window's Enter presses the button that shows the ring: the fold opens and shuts, "Show all" opens and "Show
    fewer" shuts, "Check again" reads, "Go to the run" moves. In the name, in a number and on a tick box it does nothing."""
    from PySide6.QtTest import QTest
    from tests._fixtures import whole_program
    from tests._look_helpers import _arrive
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (app, window):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        page = build_step_page(window=window)
        shown_off_screen(window, 900, 600)
        window.go_to("inference", "training")
        page.said = (Fact("warn", "a"), Fact("info", "b"))
        page.names.name.setText("retrain_3")
        page.redraw()
        _settle()
        with in_front(window):
            def ring_on(control):
                control.setFocus(Qt.TabFocusReason)
                _arrive(control, Qt.TabFocusReason)
                assert style.current_follower().target() is control

            def press(button):
                ring_on(button)
                QTest.keyClick(button, Qt.Key_Return)

            fold = page.form.fold
            press(fold.button)
            assert fold.is_open()
            press(fold.button)
            assert not fold.is_open()
            press(page.foot.more)
            assert page.foot.more.text() == "Show fewer"
            press(page.foot.more)
            assert page.foot.more.text() == "Show all 2"
            readings = page.readings
            press(page.foot.check_again)
            assert page.readings == readings + 1
            fold.set_open(True)
            for control in (page.names.name, page.form.rows["num_runs"].widget,
                            page.form.rows["run_size_cap"].widget.ticked_box):
                ring_on(control)
                ticked = control.isChecked() if hasattr(control, "isChecked") else None
                QTest.keyClick(control, Qt.Key_Return)
                assert page.pressed == 0 and window.current_place() == ("inference", "training"), type(control)
                assert ticked is None or control.isChecked() == ticked, "Enter set a tick box"
            stand_in = StandIn(word="Reduction")
            start_run(window.part("reduction"), stand_in)
            _settle()
            press(page.foot.run_link)
            assert window.current_place() == ("reduction", None)
            stand_in.release()
            turn_until(app, lambda: not controller().is_live())
        window.hide()


@pytest.mark.parametrize("size", [(900, 600), (1300, 820)], ids=str)
def test_the_tab_key_walks_a_step_in_the_order_it_is_read(size, tmp_path):
    """With real keys: the heading's buttons, the main settings, the fold and what it holds, the block for the run, what
    scrolls while it scrolls, the foot, then the rail, with an "Information." among the facts, so that the low window keeps
    its link. A row added after the page was built takes its place all the same, a greyed field is passed over for its
    box, and no help mark is a stop. No stop comes before the page's first setting but the heading's own. That each stop
    stands whole in view is held by the test of both laps, at both text sizes: at its text size the toolkit's own scroll
    on the Tab key keeps a stop in view, so a check here would see no fault of the page's."""
    window, page, _clock = _a_page(tmp_path, size=size)
    form, foot = page.form, page.foot
    form.fold.set_open(True)
    page.said = (*_facts(8, "warn"), Fact("info", "Read this too."))
    page.redraw()
    _settle()
    rows = form.rows
    expected = [page.help_button, page.head_button, rows["num_runs"].widget, rows["chi_mode"].widget, form.fold.button,
                rows["learning_rate"].widget, rows["run_size_cap"].widget.ticked_box,
                rows["max_num_epochs"].widget.ticked_box, rows["name"].widget, rows["note"].widget]
    if page.scroller.verticalScrollBar().maximum() > 0:
        expected.append(page.scroller)
    expected += [foot.facts.scroller, *([] if foot.more.isHidden() else [foot.more]), foot.check_again, foot.main_button]
    assert size[1] > 600 or not foot.more.isHidden(), "the low window lost its link"
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, len(expected))
        assert walked == expected, [type(widget).__name__ for widget in walked]
        assert QApplication.focusWidget() is window.shell.rail.row("training"), "the walk did not go on to the rail"


def _driven(page) -> list:
    """The fields of the first step's frequency response, in their order."""
    return [page.form.rows[key].widget for key in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles")]


def _applied(page) -> None:
    """The first step applied as it opens."""
    page.window().part("inference").config_panel.foot.main_button.click()


def _to_build(page, *, typed: bool = False) -> None:
    """The second step: the first applied, a new prior to build from the master bounds file, named; with ``typed``, its
    limits typed by hand."""
    _applied(page)
    page.show_view("build")
    page.bounds_picker.restore_key("master.txt")
    page.names.name.setText("retrain_prior")
    page.typed_tick.setChecked(typed)


def _knobs(page) -> list:
    """The fields of the seven settings of a build, in their order, a ticked one by its box."""
    rows = page.form.rows
    return [rows["num_iterations"].widget, rows["sweep_batch"].widget.ticked_box, rows["max_sets"].widget,
            rows["walk_step"].widget, rows["stability_units"].widget, rows["min_cluster_size"].widget,
            rows["min_samples"].widget]


def _typed_limits(page) -> list:
    """The fields of the limits typed by hand, each parameter's lower then its upper, in the file's order."""
    return [field for row in page.bounds_grid._rows.values() for field in (row.lo, row.hi)]


def _to_train(page, *, posterior: bool = False) -> None:
    """The third step: the first applied, a configuration of the master bounds file and a prior in use, a name typed; with
    ``posterior``, a posterior held too, the rotation off and the simulations not saved."""
    from types import SimpleNamespace
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, prior_stub
    _applied(page)
    screen = page.window().part("inference")
    cfg = _nad_cfg()
    cfg.reparam_rotate = not posterior
    screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior_stub(),
                                posterior=SimpleNamespace(kind="posterior", id="q1", name="retrain") if posterior
                                else None)
    screen.refresh_gates()
    page.names.name.setText("retrain")
    if posterior:
        page.form.rows["checkpoint_every"].widget.ticked_box.setChecked(True)
    page.read_facts()


def _trained(page, *, posterior: bool = False) -> tuple:
    """The fields of the third step's main settings, and of its fold, a ticked one by its box and a greyed one left out,
    in their order."""
    rows = page.form.rows
    main = [rows["num_runs"].widget, rows["run_size_cap"].widget.ticked_box, rows["hidden_features"].widget,
            rows["num_transforms"].widget, *([] if posterior else [page.resume_choice])]
    folded = [rows["learning_rate"].widget, rows["stop_after_epochs"].widget, rows["max_num_epochs"].widget.ticked_box,
              rows["expected_epochs"].widget,
              *([rows["checkpoint_every"].widget.ticked_box] if posterior else
                [rows["checkpoint_every"].widget.field, rows["checkpoint_every"].widget.ticked_box]),
              *([] if posterior else [rows[key].widget for key in ("fisher_m", "fisher_dz", "fisher_points")])]
    return main, folded


def _to_load(page) -> None:
    """The load view, with no posterior saved: the list to choose one in, and nothing to load."""
    _to_train(page)
    page.show_view("load")
    page.read_facts()


#: The third step's shapes, as ``WALKED`` reads them.
WALKED_TRAINING = {
    "a training to start": (
        _to_train, lambda page: [page.help_button, page.switch_button], lambda page: _trained(page)[0],
        lambda page: _trained(page)[1], lambda page: [page.names.name, page.names.note],
        lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again, page.foot.main_button]),
    "a posterior held, the rotation off, the simulations not saved": (
        lambda page: _to_train(page, posterior=True), lambda page: [page.help_button, page.switch_button,
                                                                    page.next_button],
        lambda page: _trained(page, posterior=True)[0], lambda page: _trained(page, posterior=True)[1],
        lambda page: [page.names.name, page.names.note],
        lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again, page.foot.main_button]),
    "a saved posterior to load": (
        _to_load, lambda page: [page.help_button, page.switch_button], lambda page: [page.post_picker.combo],
        lambda page: None, lambda page: [],
        lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again]),
}


def _a_plan_shown(page) -> None:
    """The first step with a plan of its model shown in the list and nothing filled: the plans folder the window lists,
    with one plan in the model's folder."""
    from core.gui import plans
    folder = plans.model_folder(page.window().part("inference").plans_folder, "nadrowski")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "walked.txt").write_text("# Training\nnum_runs = 7000\n", encoding="utf-8")
    page.plan_choice.showPopup()
    page.plan_choice.hidePopup()
    page.plan_choice.setCurrentIndex(page.plan_choice.findData("walked"))


#: The rebuilt pages of the window, by the step each stands on, and the shapes its walk is read in: by the name of the
#: shape, what is done to the page after its first show; the heading's buttons and the way to the next step that stand
#: in that shape; the controls of its main settings; those of the fold, while it is open, or None for a shape with no
#: fold; those of the block for the run; and the foot's, in the low window or not; each in their order. All are written
#: out by hand, never read off the page: a control the Tab key skips is then missing from the walk and not from what it
#: is held to.
WALKED = {
    "model": {
        "set by the bounds file": (lambda page: None, lambda page: [page.help_button],
                                   lambda page: [page.model_combo, page.plan_choice, page.fill_button,
                                                 page.kind_choice],
                                   lambda page: [page.rot_choice, page.units_choice], lambda page: [],
                                   lambda page, small: [*([page.foot.more] if small else []), page.foot.main_button]),
        "the frequency response, typed units, applied": (
            lambda page: (page.kind_choice.setCurrentIndex(page.kind_choice.findData(True)),
                          page.units_choice.setCurrentIndex(page.units_choice.findData("typed")),
                          page.units_text.setText("kHz ms nm pN"), page.foot.main_button.click()),
            lambda page: [page.help_button, page.next_button],
            lambda page: [page.model_combo, page.plan_choice, page.fill_button, page.save_button, page.kind_choice],
            lambda page: [*_driven(page), page.rot_choice, page.units_choice, page.units_text], lambda page: [],
            lambda page, small: [*([page.foot.more] if small else [])]),
        "a plan shown, its folder there": (
            _a_plan_shown, lambda page: [page.help_button],
            lambda page: [page.model_combo, page.plan_choice, page.holds_button, page.fill_button, page.open_button,
                          page.kind_choice],
            lambda page: [page.rot_choice, page.units_choice], lambda page: [],
            lambda page, small: [*([page.foot.more] if small else []), page.foot.main_button]),
    },
    "prior": {
        "a saved prior to load": (
            lambda page: (_applied(page), page.show_view("saved")),
            lambda page: [page.help_button, page.switch_button],
            lambda page: [page.bounds_picker.combo, page.prior_picker.combo], lambda page: None, lambda page: [],
            lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again]),
        "a new prior to build": (
            _to_build, lambda page: [page.help_button, page.switch_button], lambda page: [page.bounds_picker.combo],
            lambda page: [page.typed_tick, *_knobs(page)], lambda page: [page.names.name, page.names.note],
            lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again, page.foot.main_button]),
        "limits typed by hand": (
            lambda page: _to_build(page, typed=True), lambda page: [page.help_button, page.switch_button],
            lambda page: [page.bounds_picker.combo], lambda page: [page.typed_tick, *_typed_limits(page), *_knobs(page)],
            lambda page: [page.names.name, page.names.note],
            lambda page, small: [*([page.foot.more] if small else []), page.foot.check_again, page.foot.main_button]),
    },
    "training": WALKED_TRAINING,
}


@pytest.mark.parametrize("fold_open", [False, True], ids=["fold shut", "fold open"])
@pytest.mark.parametrize("step, shape", [(step, shape) for step, shapes in WALKED.items() for shape in shapes])
@pytest.mark.parametrize("factor", [1.0, 1.5])
@pytest.mark.parametrize("size", [(900, 600), (1300, 820)], ids=str)
def test_tab_walks_the_page_in_view_at_every_size(size, factor, step, shape, fold_open, tmp_path, monkeypatch):
    """From the step's row of the rail the Tab key goes into the page and, with real keys, walks it as it is read: the
    heading's button, then the way to the next step once there is one, then the first setting, before any button that
    starts or applies anything; every field in reading order, the fold and what it holds, a row the page shows only for
    a choice made after its first show in its place among them; what scrolls; the foot; and back to the row of the rail
    it began from. Each stop in the scrolling body stands whole in its viewport as it takes the keyboard, at the window's
    least size and a larger one, at both text sizes. The window lists the test's own folder of plans."""
    from PySide6.QtTest import QTest
    from core.gui import plans
    monkeypatch.setattr(plans, "PLAN_FOLDER", tmp_path / "plans")
    with text_size(factor), _the_look(False):
        fonts.load_app_font(qt_app())
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        design.set_room((3400, 1380))
        window.setMinimumSize(1, 1)
        shown_off_screen(window, design.scaled(size[0]), design.scaled(size[1]))
        window.go_to("inference", step)
        page = window.part("inference").pages.currentWidget()
        _settle()
        shaped, head, main, folded, block, at_the_foot = WALKED[step][shape]
        shaped(page)
        page.form.fold.set_open(fold_open)
        _settle()
        form, foot = page.form, page.foot
        rows = folded(page)
        expected = [*head(page), *main(page), *([] if rows is None else [form.fold.button, *(rows if fold_open else [])]),
                    *block(page)]
        if page.scroller.verticalScrollBar().maximum() > 0:
            expected.append(page.scroller)
        if not foot.facts.isHidden() and foot.facts.scroller.verticalScrollBar().maximum() > 0:
            expected.append(foot.facts.scroller)
        expected += at_the_foot(page, size == (900, 600))
        rail = window.shell.rail.row(step)
        with in_front(window):
            rail.setFocus(Qt.TabFocusReason)
            _settle()
            walked, out = [], []
            for _ in range(len(expected) + 1):
                QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
                _settle()
                walked.append(QApplication.focusWidget())
                if page.scroller.body.isAncestorOf(walked[-1]) and not _in_view(page, walked[-1]):
                    out.append(type(walked[-1]).__name__)
        assert walked == [*expected, rail], [type(widget).__name__ for widget in walked]
        assert out == [], out
        first = next(at for at, stop in enumerate(walked) if form.isAncestorOf(stop))
        assert walked[first] is page.first_focus(), "the walk reached a setting before the page's first"
        assert not any(stop in walked[:first] for stop in foot.stops()), "a button that applies came before the form"
        inside = page.form.fold.stops()[1:]
        assert [stop for stop in walked if stop in inside] == (rows if rows is not None and fold_open else []), (
            "the fold's rows that stand are not the shape's")


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_widest_model_name_a_person_can_save_fits_the_first_step_at_the_least_size(factor, tmp_path):
    """A model of your own is named in at most 24 letters and figures, and the list of models is as wide as its longest
    item: the widest name, chosen, leaves the form inside its viewport at the window's least size with the fold open,
    for the body never scrolls sideways."""
    from core import registry
    from core.Helpers import model_store
    name = "W" * 24
    assert model_store.NAME_RE.match(name) and not model_store.NAME_RE.match(name + "W"), "the longest name moved"
    registry.register(registry.ModelSpec(name, ["a"], is_user_model=True, n_vars=1))
    try:
        with text_size(factor), _the_look(False):
            fonts.load_app_font(qt_app())
            controller_on(tmp_path, Clock(at(8, 12, 4)))
            window = build_window()
            design.set_room((3400, 1380))
            window.setMinimumSize(1, 1)
            page = window.part("inference").config_panel
            page.model_combo.setCurrentText(name)
            page.kind_choice.setCurrentIndex(page.kind_choice.findData(True))
            page.form.fold.set_open(True)
            shown_off_screen(window, design.scaled(900), design.scaled(600))
            window.go_to("inference", "model")
            _settle()
            port = page.scroller.viewport()
            assert page.model_combo.width() >= page.model_combo.minimumSizeHint().width()
            assert page.scroller.body.minimumSizeHint().width() <= port.width(), (
                page.scroller.body.minimumSizeHint().width(), port.width())
            assert _edge_of(page.model_combo, port).right() < port.width()
    finally:
        registry.unregister(name)


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_longest_file_name_fits_the_second_step_at_the_least_size(factor, tmp_path):
    """A bounds file is named by a person, at any length. The list of bounds files is as wide as its longest name up to a
    cap, past which the name shown is cut short and stands whole in the tip and in the list as it opens: chosen, an 86
    letters' name leaves the form inside its viewport at the window's least size with the fold open and the limits typed
    by hand, for the body never scrolls sideways."""
    name = "a_bounds_file_named_at_length_for_the_cell_measured_on_the_second_of_november_2026.txt"
    assert len(name) == 86
    with text_size(factor), _the_look(False):
        fonts.load_app_font(qt_app())
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        design.set_room((3400, 1380))
        window.setMinimumSize(1, 1)
        inference = window.part("inference")
        inference.config_panel.foot.main_button.click()
        page = inference.prior_panel
        (tmp_path / name).write_text((Path(page.bounds_picker.base_path) / "master.txt").read_text(encoding="utf-8"),
                                     encoding="utf-8")
        page.bounds_picker.repoint(tmp_path, name)
        shown_off_screen(window, design.scaled(900), design.scaled(600))
        window.go_to("inference", "prior")
        _settle()
        page.show_view("build")
        page.typed_tick.setChecked(True)
        page.form.fold.set_open(True)
        _settle()
        combo, port = page.bounds_picker.combo, page.scroller.viewport()
        assert combo.currentText() == name and combo.toolTip() == name
        assert combo.width() >= combo.minimumSizeHint().width()
        assert page.scroller.body.minimumSizeHint().width() <= port.width(), (
            page.scroller.body.minimumSizeHint().width(), port.width())
        assert _edge_of(combo, port).right() < port.width()
        combo.showPopup()
        try:
            assert combo.view().minimumWidth() > combo.width() and combo.view().minimumWidth() >= (
                combo.fontMetrics().horizontalAdvance(name))
        finally:
            combo.hidePopup()


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_every_steps_body_stands_within_its_view_at_the_least_size(factor, tmp_path):
    """At the window's least size, at the ordinary text size and at one and a half times it, the body of each step a
    plan fills -- Model and options, Prior and Training -- stands within its view: it asks no more width than the view
    gives, and no part of it stands past the view's right edge. The Training step's longest choice of "Continue or
    start new", chosen, is held to the room its row has where it must be, cut short with the whole of it in the tip."""
    with text_size(factor), _the_look(False):
        fonts.load_app_font(qt_app())
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        design.set_room((3400, 1380))
        window.setMinimumSize(1, 1)
        shown_off_screen(window, design.scaled(900), design.scaled(600))
        inference = window.part("inference")
        training = inference.posterior_panel
        choice = training.resume_choice
        choice.setCurrentIndex(choice.findData("never"))
        for step, page in (("model", inference.config_panel), ("prior", inference.prior_panel), ("training", training)):
            window.go_to("inference", step)
            _settle()
            port, body = page.scroller.viewport(), page.scroller.body
            assert body.minimumSizeHint().width() <= port.width(), (step, body.minimumSizeHint().width(), port.width())
            out = [(type(part).__name__, _edge_of(part, port).right()) for part in [body, *body.findChildren(QWidget)]
                   if part.isVisibleTo(page) and _edge_of(part, port).right() >= port.width()]
            assert out == [], (step, port.width(), out[:6])
        if choice.width() < choice.fontMetrics().horizontalAdvance(choice.currentText()):
            assert choice.toolTip() == choice.currentText(), "a choice cut short has not the whole of it in its tip"


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_the_keyboard_lands_in_view_on_both_laps(factor, tmp_path):
    """Every control of the body the Tab key stops at stands whole in the viewport, at the window's least size and both text
    sizes, on the lap that starts at the top and on the next, which comes back into the body from outside it: the toolkit
    brings a stop into view only when the keyboard leaves another inside the same area, and then only as far as a text
    box's caret."""
    from PySide6.QtTest import QTest
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        shown_off_screen(window, design.scaled(900), design.scaled(600))
        window.go_to("inference", "training")
        page.form.fold.set_open(True)
        page.said = _facts(9, "warn")
        page.redraw()
        _settle()
        first, body = page.form.rows["num_runs"].widget, page.scroller.body
        assert page.scroller.verticalScrollBar().maximum() > 0, "nothing scrolls, so this says nothing"
        with in_front(window):
            page.help_button.setFocus(Qt.TabFocusReason)
            laps, stops, out = 0, 0, []
            for _ in range(80):
                QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
                _settle()
                found = QApplication.focusWidget()
                laps += found is first
                if laps == 3:
                    break
                if body.isAncestorOf(found):
                    stops += 1
                    if not _in_view(page, found):
                        out.append(type(found).__name__)
            assert laps == 3 and stops == 16 and out == [], (laps, stops, out)


def test_a_tall_part_shows_its_top(tmp_path):
    """A control higher than the viewport cannot stand whole in it, so it stands with its top in view."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    tall = page.form.rows["chi_mode"].widget
    tall.setFixedHeight(page.scroller.viewport().height() * 2)
    _settle()
    page.scroller.verticalScrollBar().setValue(0)
    page.scroller.bring_into_view(tall)
    assert _edge_of(tall, page.scroller.viewport()).top() >= 0


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_how_the_run_went_takes_the_keyboard_in_view(factor, tmp_path):
    """A hand-over is a change of focus like any other: a run started from a form scrolled to its foot, by the keyboard,
    ends with the keyboard on "How the run went", at the top of the body."""
    app = qt_app()
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        shown_off_screen(window, design.scaled(900), design.scaled(600))
        window.go_to("inference", "training")
        page.form.fold.set_open(True)
        page.said = _facts(9, "warn")
        page.redraw()
        _settle()
        bar = page.scroller.verticalScrollBar()
        bar.setValue(bar.maximum())
        stand_in = StandIn()
        page.main_pressed = lambda: _start(page, stand_in)
        with in_front(window):
            page.foot.main_button.setFocus(Qt.TabFocusReason)
            page.foot.main_button.click()
            assert QApplication.focusWidget() is page.run_view.stop_button
            stand_in.release()
            turn_until(app, lambda: not controller().is_live())
            _settle()
            assert QApplication.focusWidget() is page.run_view.how_link and bar.value() < bar.maximum()
            assert _in_view(page, page.run_view.how_link)


def test_the_keyboard_scrolls_no_page_out_of_sight(tmp_path):
    """The change of focus is heard from the whole program, and a page not in sight stays as it was scrolled."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    page.form.fold.set_open(True)
    page.said = _facts(9, "warn")
    page.redraw()
    _settle()
    bar = page.scroller.verticalScrollBar()
    bar.setValue(bar.maximum())
    place = bar.value()
    window.go_to("inference", "prior")
    _settle()
    with in_front(window):
        page.form.rows["num_runs"].widget.setFocus(Qt.OtherFocusReason)
        _settle()
        assert bar.value() == place


def test_the_keyboard_outside_the_body_scrolls_nothing(tmp_path):
    """The heading and the foot stand outside what scrolls, so a control there needs no bringing into view."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    page.form.fold.set_open(True)
    page.said = _facts(9, "warn")
    page.redraw()
    _settle()
    bar = page.scroller.verticalScrollBar()
    with in_front(window):
        for place, control in ((bar.maximum(), page.help_button), (bar.maximum() // 2, page.foot.main_button)):
            page.form.rows["learning_rate"].widget.setFocus(Qt.TabFocusReason)
            _settle()
            bar.setValue(place)
            control.setFocus(Qt.TabFocusReason)
            _settle()
            assert bar.value() == place, type(control).__name__


def test_a_window_that_goes_to_the_back_and_comes_to_the_front_stays_as_it_was_scrolled(tmp_path):
    """A window that goes to the back takes the keyboard from its control to nowhere; brought to the front again it hands
    it back from nowhere; a box or a figure's window that opens takes it from the control, and one that closes hands it
    back from nowhere or from that window; and a move between two controls of such a window stays in it. None of them
    moved the keyboard on the page: a person who scrolled away from the control with the wheel finds the page where they
    left it."""
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    page.form.fold.set_open(True)
    page.said = _facts(9, "warn")
    page.redraw()
    _settle()
    field = page.form.rows["num_runs"].widget
    bar = page.scroller.verticalScrollBar()
    with in_front(window):
        field.setFocus(Qt.TabFocusReason)
        bar.setValue(bar.maximum())
        _settle()
        place = bar.value()
        assert place > 0 and not _in_view(page, field), "the field stands in view, so this says nothing"
        elsewhere = build_part(QPushButton)
        other = build_part(QWidget)
        one, two = QPushButton(other), QPushButton(other)
        for before, now in ((field, None), (None, field), (elsewhere, field), (field, elsewhere), (one, two)):
            page._focus_moved(before, now)
            _settle()
            assert bar.value() == place, (before, now)
    with in_front(window):
        _settle()
        assert QApplication.focusWidget() is field and bar.value() == place, "the round trip moved the page"


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_a_refusal_of_the_field_that_holds_the_keyboard_brings_it_into_view(factor, tmp_path):
    """The keyboard does not move, so no change of focus is heard: the refusal asks for itself."""
    from core.refusals import Refusal
    with text_size(factor), _the_look(False):
        window, page, _clock = _a_page(tmp_path, shown=False)
        shown_off_screen(window, design.scaled(900), design.scaled(600))
        window.go_to("inference", "training")
        page.form.fold.set_open(True)
        page.said = _facts(9, "warn")
        page.redraw()
        _settle()
        field = page.form.rows["num_runs"].widget
        bar = page.scroller.verticalScrollBar()
        with in_front(window):
            field.setFocus(Qt.TabFocusReason)
            _settle()
            bar.setValue(bar.maximum())
            _settle()
            assert not _in_view(page, field), "the field stands in view, so this says nothing"
            page._refusal(Refusal("No.", field="num_runs"))
            _settle()
            assert _in_view(page, field)


@pytest.mark.parametrize("programs_font", [True, False], ids=["the program's font", "the toolkit's"])
def test_a_refusal_that_grows_the_foot_leaves_its_field_in_view(programs_font, tmp_path):
    """The refusal adds a fact, so the foot grows and the viewport shrinks in the same turn as the keyboard moves: the last
    field of the form still stands whole in what is left, whichever turn the layouts settle in, which the font decides."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path, size=(900, 600), programs_font=programs_font)
    page.form.fold.set_open(True)
    _settle()
    before = page.scroller.viewport().height()
    note = page.form.rows["note"]
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        page._refusal(Refusal("A sentence that takes two lines and more at this width, for the foot has to grow by "
                              "something a person could read; it says why the note cannot be used, and what to do.",
                              field="note"))
        _settle()
        assert page.scroller.viewport().height() < before, "the foot did not grow, so this says nothing"
        assert QApplication.focusWidget() is note.widget and _in_view(page, note.widget)


def test_a_move_to_a_step_brings_no_other_step_into_sight(tmp_path):
    """Going to one step from another part of the window reads that step, and not the one last shown there, which a person
    never sees."""
    window, page, _clock = _a_page(tmp_path)
    window.go_to("home")
    _settle()
    page.stale = True
    window.go_to("inference", "prior")
    _settle()
    assert page.readings == 0 and page.stale, "a step nobody sees came into sight on the way"


def test_the_walk_takes_the_figures_where_they_are_read(tmp_path):
    """After a run with a figure: "How the run went", the figure's tab, its "Pop out" and what scrolls it, the next step's
    button, then the form, and from the foot to the rail. The figure came after the first show, so it joins the order as
    it comes."""
    from PySide6.QtTest import QTest
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in, provide_fig_sink=True)
    stand_in.figure("A figure")
    turn_until(app, lambda: page.figure_stack.count() == 1)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    page.set_next("Go to Checks", "checks")
    _settle()
    figure = page.figure_stack.widget(0)
    expected = [page.help_button, page.head_button, page.run_view.how_link, page.figure_stack.tabBar(), figure._pop,
                figure.findChildren(QScrollArea)[0], page.next_button, page.form.rows["num_runs"].widget]
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, len(expected))
        assert walked == expected, [type(widget).__name__ for widget in walked]
        page.foot.main_button.setFocus(Qt.TabFocusReason)
        QTest.keyClick(page.foot.main_button, Qt.Key_Tab)
        _settle()
        assert QApplication.focusWidget() is window.shell.rail.row("training"), "the walk did not go on to the rail"


def test_the_walk_on_a_runs_page_reaches_every_button_and_the_figure_and_goes_on(tmp_path):
    """The heading's buttons, the run's own, the figure's, then the rail: nothing of the page falls out of the order, and
    the keyboard does not go round the figure for ever."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in, provide_fig_sink=True)
    stand_in.figure("A figure")
    turn_until(app, lambda: page.figure_stack.count() == 1)
    _settle()
    figure = page.figure_stack.widget(0)
    expected = [page.help_button, page.head_button, page.run_view.text_button, page.run_view.stop_button,
                page.figure_stack.tabBar(), figure._pop, figure.findChildren(QScrollArea)[0], window.shell.rail.row("training")]
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, len(expected))
    assert walked == expected, [type(widget).__name__ for widget in walked]
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


def test_the_walk_during_a_run_on_another_page_reaches_the_way_to_it(tmp_path):
    """While a run locks the form the keyboard goes from the heading to the facts, the link that shows the rest, "Go to the
    run" and the rail: the two buttons are greyed."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path, size=(900, 600))
    foot = page.foot
    page.said = (*_facts(8, "warn"), Fact("info", "Read this too."))
    page.redraw()
    _settle()
    stand_in = StandIn(word="Reduction")
    start_run(window.part("reduction"), stand_in)
    window.go_to("inference", "training")
    _settle()
    expected = [page.help_button, page.head_button, page.scroller, foot.facts.scroller, foot.more, foot.run_link]
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, len(expected))
        assert walked == expected, [type(widget).__name__ for widget in walked]
        assert QApplication.focusWidget() is window.shell.rail.row("training")
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())


def test_a_row_added_after_the_first_show_takes_its_place_in_the_walk(tmp_path):
    window, page, _clock = _a_page(tmp_path)
    late = IntField(1)
    page.form.add("n_cal", late, usual=1)
    _settle()
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, 5)
    assert walked[2:5] == [page.form.rows["num_runs"].widget, page.kind_choice, late]


def test_a_control_that_makes_its_fields_again_says_so_and_they_take_their_place(tmp_path):
    """A grid of fields, made again after the first show, as the bounds' is when the file changes: the page cannot hear its
    children come, so it says so by ``order_the_keys``, and the fields stand where the grid stands."""
    from PySide6.QtWidgets import QVBoxLayout

    class Grid(QWidget):
        def __init__(self):
            super().__init__()
            self.setLayout(QVBoxLayout())
            self.fields = []

        def load(self, count):
            for field in self.fields:
                self.layout().removeWidget(field)
                field.deleteLater()
            self.fields = [QLineEdit() for _ in range(count)]
            for field in self.fields:
                self.layout().addWidget(field)

    window, page, _clock = _a_page(tmp_path)
    grid = Grid()
    grid.load(1)
    page.form.add("n_cal", grid, label="Bounds")
    _settle()
    gone = grid.fields
    grid.load(3)
    QApplication.sendPostedEvents(gone[0], QEvent.Type.DeferredDelete)
    _settle()
    page.order_the_keys()
    with in_front(window):
        page.help_button.setFocus(Qt.TabFocusReason)
        walked = _walked(window, 7)
    assert walked[4:7] == grid.fields, [type(widget).__name__ for widget in walked]


def test_access_keys_stand_only_on_the_main_button_and_the_heading(tmp_path):
    """On the page's form and its foot's other controls no word carries an access key; the main button and the heading's
    buttons do."""
    window, page, _clock = _a_page(tmp_path)
    keyed = [button for button in page.findChildren(QPushButton) if re.search(r"&[^&]", button.text())
             and not page.run_view.isAncestorOf(button)]
    assert set(keyed) == {page.foot.main_button, page.help_button, page.head_button}, [b.text() for b in keyed]
    assert not [label for label in page.findChildren(QLabel) if label.buddy() is not None and "&" in label.text()]


def test_a_refusal_marks_its_row_and_gives_it_the_keyboard(tmp_path):
    """Refused at the press or by the stage: the shared question, telling, with where to fix it said from this page; when
    it closes the row keeps the engine's sentence under the field and takes the keyboard, and the first edit takes the
    sentence away. A folded row opens its fold for it."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    field = page.form.rows["num_runs"].widget
    with in_front(window):
        page._refusal(Refusal("The number of simulation batches must be at least 1; got 0.", field="num_runs"))
        said = asked()
        assert (said.title, said.sentence, said.beneath, said.answers, said.marker) == (
            "Check your inputs", "The number of simulation batches must be at least 1; got 0.",
            "Set it in 'Batches' on this page.", ("OK",), "warn")
        row = page.form.rows["num_runs"]
        assert row.line.label.text() == "The number of simulation batches must be at least 1; got 0."
        assert field.error() == row.line.label.text() and QApplication.focusWidget() is field
        field.setText("3")
        assert row.line.isHidden() and field.error() == ""
        page._refusal(Refusal("The learning rate must be above 0; got 0.", field="learning_rate"))
        rate = page.form.rows["learning_rate"]
        assert page.form.fold.is_open() and QApplication.focusWidget() is rate.widget
        window.resize(900, 600)
        page.said = _facts(9, "warn")
        page.redraw()
        _settle()
        page.scroller.verticalScrollBar().setValue(0)
        page._refusal(Refusal("The note cannot be used.", field="note"))
        note = page.form.rows["note"]
        assert QApplication.focusWidget() is note.widget and note.line.label.text() == "The note cannot be used."
        _settle()
        assert page.scroller.verticalScrollBar().value() > 0, "the row stood in sight already, so this says nothing"
        rectangle = _edge_of(note.widget, page.scroller.viewport())
        assert page.scroller.viewport().rect().contains(rectangle), "the row is out of sight"


def test_a_refusal_with_no_row_stands_at_the_head_of_the_facts_until_they_are_read(tmp_path):
    """It is an "Error." row, first of all, that says where to fix it, and greys the button with its reason, until the facts
    are next read; a hidden row is no row."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    page.names.name.setText("retrain_3")
    page.names._store = _Store()
    page.names.check_now()
    page.said = (Fact("info", "Another fact."),)
    page.redraw()
    assert page.foot.main_button.isEnabled()
    page._refusal(Refusal("The simulation set cannot be used.", field="new_run"))
    row = page.foot.facts.rows()[0]
    assert (row.kind, row.sentence.plain_text()) == (
        "err", "Error. The simulation set cannot be used. Answer the question that the start button asks on this page.")
    assert not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: the last start was refused."
    page.foot.check_again.click()
    assert [row.kind for row in page.foot.facts.rows()] == ["info"] and page.foot.main_button.isEnabled()
    page.form.rows["learning_rate"].set_shown(False)
    page._refusal(Refusal("No.", field="learning_rate"))
    assert page.foot.facts.rows()[0].kind == "err" and not page.form.fold.is_open()
    page.foot.check_again.click()
    page._refusal(Refusal("Fixed by measurement.", field="chi_f0"))
    assert page.foot.facts.rows()[0].sentence.plain_text().startswith("Error. Fixed by measurement.")
    assert page.form.rows["chi_f0"].line.isHidden(), "a sentence stood under a value nothing can edit"


def test_a_refusals_sentence_stands_through_a_pages_own_lines_until_its_field_is_edited(tmp_path):
    """"Batches" has a live rule, said under it at every drawing, as every ruled row of a page has: it says nothing over a
    refusal's sentence, at an edit of another field or at a drawing, and the first edit of the field takes the sentence away
    and the rule's line comes at that same drawing. A page's line with words waits as an empty one does."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    row = page.form.rows["num_runs"]
    sentence = "The number of simulation batches must be at least 1; got 0."
    page._refusal(Refusal(sentence, field="num_runs"))
    for moved in (lambda: page.kind_choice.setCurrentIndex(1), page.redraw, lambda: row.set_line("warn", "A line.")):
        moved()
        assert (row.line.label.text(), row.line.isHidden(), row.in_error()) == (sentence, False, True)
    row.widget.setText("")
    assert row.line.label.text() == "Batches is blank." and row.in_error()
    row.widget.setText("6")
    assert row.line.isHidden() and not row.in_error()


def test_a_marked_row_stops_the_start_until_its_field_is_edited(tmp_path):
    """A row in error never stands beside a lit button: the refusal is an "Error." among the facts, in the words of a
    refusal with no row, and the first edit of the field ends it."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    foot, row = page.foot, page.form.rows["num_runs"]
    assert foot.main_button.isEnabled()
    page._refusal(Refusal("Too many.", field="num_runs"))
    assert not foot.main_button.isEnabled() and foot.reason.text() == "Cannot start yet: the last start was refused."
    assert [(fact.kind, fact.sentence.plain_text()) for fact in foot.facts.rows()] == [("err", "Error. Too many.")]
    row.widget.setText("")
    assert [(fact.kind, fact.sentence.plain_text()) for fact in foot.facts.rows()] == [("err", "Error. Batches is blank.")]
    assert foot.reason.text() == "Cannot start yet: a setting is not a number.", "the refusal outlived its mark"
    row.widget.setText("6")
    assert foot.main_button.isEnabled() and foot.reason.text() == "" and foot.facts.rows() == []


def _a_page_with_nothing_to_read():
    """The stand-in page as a step with nothing saved to read: no "read at" line, no "Check again"."""
    from tests._fixtures import _the_stand_in_page

    class Quiet(_the_stand_in_page()):
        reads_the_disk = False

    return Quiet


@pytest.mark.parametrize("how", ["an edit of another field", "an edit on a page with nothing to read", "Check again",
                                 "a change of the store", "the next show"])
def test_a_reading_ends_what_a_refusal_stops_and_its_sentence_stands(how, tmp_path):
    """What cures a refusal may stand in another field, or on another page, and a person who mended it there must not have
    to touch the field that was named. So the facts read again end what a refusal stops, as they end one with no row, and the
    press asks the engine again; on a page with nothing saved to read, and so no "Check again", the edit asks for that
    reading itself. The sentence stands under its field until the field is edited, no longer an error, for a marker never
    disagrees with the button; a rule's empty line still leaves it, and a line with words takes its place."""
    from core.refusals import Refusal
    quiet = how == "an edit on a page with nothing to read"
    window, page, _clock = _a_page(tmp_path, cls=_a_page_with_nothing_to_read() if quiet else None)
    foot, row = page.foot, page.form.rows["num_runs"]
    assert foot.check_again.isHidden() is quiet
    page._refusal(Refusal("Must be at most the number of slots, 8; got 9.", field="num_runs"))
    assert not foot.main_button.isEnabled()
    if how.startswith("an edit"):
        page.kind_choice.setCurrentIndex(1)
        assert not foot.main_button.isEnabled(), "the edit alone is no reading"
        assert page.read_timer.isActive(), "the edit asked for no reading"
        page.read_timer.timeout.emit()
    elif how == "Check again":
        foot.check_again.click()
    elif how == "a change of the store":
        window.part("inference").stale_store()
    else:
        window.go_to("inference", "prior")
        _settle()
        page.stale = True
        window.go_to("inference", "training")
        _settle()
    assert foot.main_button.isEnabled() and foot.reason.text() == "" and foot.facts.rows() == []
    assert (row.line.label.text(), row.line.kind, row.in_error()) == (
        "Must be at most the number of slots, 8; got 9.", "warn", False)
    assert row.widget.error() == "" and row.widget.property("state") == "", "still outlined beside a lit button"
    page.redraw()
    assert row.line.label.text() == "Must be at most the number of slots, 8; got 9.", "a rule's empty line took it"
    row.set_line("warn", "A later line.")
    assert row.line.label.text() == "A later line."
    page.redraw()
    assert row.line.isHidden(), "the sentence came back, or the line stood after the mark had gone"


def test_a_pressed_start_refused_again_marks_its_row_again(tmp_path):
    """After a reading the button is lit, and a press that the engine refuses again is a refusal like the first."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    foot = page.foot
    refusal = Refusal("Too many.", field="num_runs")
    page.main_pressed = lambda: page._refusal(refusal)
    page._refusal(refusal)
    foot.check_again.click()
    assert foot.main_button.isEnabled()
    foot.main_button.click()
    assert not foot.main_button.isEnabled() and page.form.rows["num_runs"].in_error()


def test_a_name_the_stage_refused_is_read_again_with_the_facts(tmp_path):
    """A stage may refuse a name the page last read as free. The next reading says what the store says now: the sentence
    gives way to it, and a name that is free again starts."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    store = _Store(taken=("retrain_3",))
    page.names._store = store
    row = page.form.rows["name"]
    page._refusal(Refusal("The name was taken while the run was going.", field="name"))
    assert row.line.label.text() == "The name was taken while the run was going."
    page.foot.check_again.click()
    assert row.line.kind == "err" and page.foot.reason.text() == "Cannot start yet: the name cannot be used."
    store.taken = ()
    page.foot.check_again.click()
    assert (row.line.kind, row.line.label.text()) == ("fine", NAME_FREE_TEXT) and page.foot.main_button.isEnabled()


def test_a_marked_row_hidden_afterwards_stops_nothing(tmp_path):
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    row = page.form.rows["max_num_epochs"]
    page._refusal(Refusal("No.", field="max_num_epochs"))
    assert not page.foot.main_button.isEnabled()
    row.set_shown(False)
    page.redraw()
    assert page.foot.main_button.isEnabled() and page.foot.facts.rows() == []


def test_a_refusal_on_a_page_not_in_sight_marks_its_row_and_leaves_the_keyboard(tmp_path):
    """The keys stay with what a person is doing; the mark waits for them on the page."""
    from core.refusals import Refusal
    window, page, _clock = _a_page(tmp_path)
    with in_front(window):
        window.go_to("inference", "prior")
        _settle()
        held = QApplication.focusWidget()
        page._refusal(Refusal("No.", field="num_runs"))
        _settle()
        assert QApplication.focusWidget() is held
    assert page.form.rows["num_runs"].line.label.text() == "No."


@pytest.mark.parametrize("key", ["new_run", "num_runs"], ids=["no row", "a row"])
def test_a_refusal_before_anything_is_spent_comes_back_to_the_page_and_stands(key, tmp_path):
    """The stage refuses at its entry: the column comes back, the run is no run, and the page says the refusal as it says
    one at its button. The reading of the run's end comes first, so that it does not take the refusal away."""
    from core.refusals import Refusal
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    stand_in.fail(Refusal("The name is taken.", field=key))
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert asked().title == "Check your inputs"
    assert page.results.currentWidget() is page.figures
    assert page.foot.facts.rows()[0].kind == "err" and not page.foot.main_button.isEnabled()
    assert not page.read_timer.isActive(), "a reading waits that would take the refusal away"


# ── every end of a run, on a step page ──
def _a_training(tmp_path, page):
    """A training as its page hands it to its run, and what a test needs to bring it to a stage: (the run's text, the
    simulation set's folder, the spec)."""
    import torch
    from tests._runs import RunText, training_spec
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"batches_done": 200, "complete": False}, folder / "state.pt")
    return RunText(controller()), folder, training_spec(place=("inference", "training"), stop_button="Stop training…")


def _out_of_memory():
    try:
        raise RuntimeError("CUDA error: out of memory")
    except RuntimeError as card:
        try:
            raise RuntimeError("the simulation of batch 237 failed") from card
        except RuntimeError as raised:
            return raised


def test_a_training_tried_again_keeps_the_run_page_and_the_lock_until_it_ends(tmp_path):
    """After the card ran out of memory the run waits to be tried again, live: the page stays on the run's page, locked, its
    reading owed; the try goes on the same page, and its finish gives the column back and reads once."""
    from core.gui.run import retry
    from tests._runs import to_the_simulation
    app = qt_app()
    clock = Clock(at(3, 3, 2, 11))
    window, page, _clock = _a_page(tmp_path, clock=clock)
    text, folder, spec = _a_training(tmp_path, page)
    held = controller()
    stand_in = StandIn()
    _start(page, stand_in, spec=spec)
    to_the_simulation(app, stand_in, text, held, folder, 236)
    stand_in.fail(_out_of_memory())
    turn_until(app, lambda: held.status.state == "retrying")
    assert page.results.currentWidget() is page.run_view and not page.form.isEnabled() and page.readings == 0
    clock.advance(retry.PAUSES_S[0] + 1)
    held.tick()
    turn_until(app, lambda: len(stand_in.calls) == 2)
    assert page.results.currentWidget() is page.run_view and page.readings == 0
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    _settle()
    assert page.results.currentWidget() is page.figures and page.form.isEnabled() and page.readings == 1


def test_a_training_continued_from_its_page_starts_on_that_page_and_empties_its_figures(tmp_path):
    """"Continue from batch N" on a failed training's page is the page's next run: the page is locked and takes the run again,
    with the page's session the one the call was made in, and the figures of the failed try go. Its end reads once."""
    from tests._runs import to_the_simulation
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    text, folder, spec = _a_training(tmp_path, page)
    held, view = controller(), page.run_view
    stand_in = StandIn()
    _start(page, stand_in, spec=spec, name="retrain_3", num_runs=10_000)
    page.figure_stack.add_figure("A figure", b"")
    to_the_simulation(app, stand_in, text, held, folder, 236)
    stand_in.fail(OSError(28, "No space left on device"))
    turn_until(app, lambda: not held.is_live())
    _settle()
    assert view.continue_button.isVisibleTo(view) and page.results.currentWidget() is view
    assert page.form.isEnabled() and page.readings == 1
    view.continue_button.click()
    turn_until(app, lambda: len(stand_in.calls) == 2)
    assert (held.is_live(), not page.form.isEnabled(), page.figure_stack.count()) == (True, True, 0)
    assert stand_in.calls[1][1]["resume"] == "require"
    stand_in.release("the posterior")
    turn_until(app, lambda: not held.is_live())
    _settle()
    assert page.results.currentWidget() is page.figures and page.readings == 2


def test_a_continue_the_session_no_longer_allows_starts_nothing_and_says_why(tmp_path):
    """The page's session is what the call is held against: once it holds another configuration the offer is no offer."""
    from core.gui.run import retry
    from tests._runs import to_the_simulation
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    text, folder, spec = _a_training(tmp_path, page)
    held, view = controller(), page.run_view
    session = page.session
    session.cfg, session.inf_prior = object(), object()
    stand_in = StandIn()
    page.run_spec = lambda fn, *args, **kw: spec
    assert page.dispatch(stand_in, session.cfg, session.inf_prior, name="retrain_3", num_runs=10_000)
    del page.run_spec
    to_the_simulation(app, stand_in, text, held, folder, 236)
    stand_in.fail(OSError(28, "No space left on device"))
    turn_until(app, lambda: not held.is_live())
    _settle()
    assert view.continue_button.isVisibleTo(view)
    session.cfg = object()
    view.continue_button.click()
    _settle()
    live = held.is_live()
    if live:
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
    assert not live and len(stand_in.calls) == 1, "a continue started in a session that has moved on"
    assert view.notice.text() == retry.CHANGED_TEXT


def test_a_run_with_no_figure_leaves_no_empty_frame(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.figure_stack.isHidden(), "an empty frame stands in the body after a run"


def test_what_is_in_use_follows_a_runs_end(tmp_path):
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    page.held = ("retrain_prior", "Built on master.txt.")
    stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.in_use.line_text() == "In use: retrain_prior"


@pytest.mark.parametrize("how", ["a stop", "a failure"])
def test_a_stop_and_a_failure_give_the_page_back_whole(how, tmp_path):
    """The foot's two buttons and the way to the run, at the run's end; the figures in their place, at "Back to the form"."""
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    foot = page.foot
    stand_in = StandIn()
    _start(page, stand_in)
    if how == "a stop":
        controller().request_stop()
    else:
        stand_in.fail(RuntimeError("the stand-in failed"))
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert foot.main_button.isEnabled() and foot.check_again.isEnabled() and foot.run_link.isHidden()
    page.run_view.back_button.click()
    _settle()
    assert page.scroller.isAncestorOf(page.figure_stack), "the figures did not come back to their place"


def test_a_training_resumed_from_its_page_starts_on_that_page(tmp_path):
    """"Resume the training" after a stop that kept it: the page's next run, which empties the figures of the try it comes
    from, and a refusal made before anything was spent is said on the run's page, in the words of the page it stands on."""
    from core.refusals import Refusal
    from tests._dialogs import choose
    from tests._fixtures import build_fit_record
    from tests._runs import note, to_the_fit
    from core.artifacts import default_store
    app = qt_app()
    clock = Clock(at(3, 3, 2, 11))
    window, page, _clock = _a_page(tmp_path, clock=clock)
    text, folder, spec = _a_training(tmp_path, page)
    held, view = controller(), page.run_view
    kept = build_fit_record(default_store(), passes=3)
    stand_in = StandIn()
    stand_in.kept_fit = note(kept, 3)
    _start(page, stand_in, spec=spec, name="retrain_3")
    page.figure_stack.add_figure("A figure", b"")
    to_the_fit(app, stand_in, text, clock, folder, [(1, 0), (2, 1), (3, 2)])
    with pytest.MonkeyPatch.context() as patch:
        choose(patch, words="Stop training")
        view.stop_button.click()
        turn_until(app, lambda: not held.is_live())
    _settle()
    assert view.resume_button.isVisibleTo(view) and page.results.currentWidget() is view and page.form.isEnabled()
    view.resume_button.click()
    turn_until(app, lambda: len(stand_in.calls) == 2)
    assert stand_in.calls[1][1]["resume_fit"] == kept and not page.form.isEnabled()
    assert page.figure_stack.count() == 0, "the figures of the stopped try stayed through a resume"
    stand_in.fail(Refusal("The name is taken.", field="name"))
    turn_until(app, lambda: not held.is_live())
    _settle()
    assert view.sentence.text() == "The name is taken. Choose another name in 'Name' on this page."
    assert page.results.currentWidget() is view
    view.back_button.click()


# ── the fix sentence ──
def test_the_fix_sentence_names_this_page_or_the_steps_that_show_the_setting():
    """Each form of the sentence, and the table's own for every key at no page, which the old pages keep, but for the
    settings of the Training page alone, for which a page of the old kind names no box."""
    assert fields.fix_sentence("num_runs", here="training") == "Set it in 'Batches' on this page."
    assert fields.fix_sentence("num_runs", here="narrowing") == "Set it in 'Batches' on this page."
    assert fields.fix_sentence("num_runs", here="prior") == (
        "Set it in 'Batches' in step 3, Training, or step 6, Narrowing round.")
    assert fields.fix_sentence("bounds", here="training") == "Set it in 'Bounds file' in step 2, Prior."
    assert fields.fix_sentence("new_run", here="training") == "Answer the question that the start button asks on this page."
    assert fields.fix_sentence("name", here="prior") == "Choose another name in 'Name' on this page."
    assert fields.fix_sentence("note", here="prior") == "Change it in 'Note' on this page."
    assert fields.fix_sentence("artifact", here="prior") == fields.fix_sentence("artifact") != ""
    assert fields.fix_sentence("device", here="prior") == "" and fields.fix_sentence(None, here="prior") == ""
    assert fields.fix_sentence("no_such_key", here="prior") == ""
    for key, entry in fields.CONTROL.items():
        if key in fields.ON_TRAINING_ALONE:
            assert fields.fix_sentence(key) == "" and fields.fix_sentence(key, here="training") == (
                f"Set it in '{entry[1]}' on this page."), key
        elif isinstance(entry, tuple):
            assert fields.fix_sentence(key) == f"Set it in the '{entry[1]}' box {fields._in(entry[0])}.", key
        else:
            assert fields.fix_sentence(key) == (entry or ""), key
        assert isinstance(fields.fix_sentence(key, here="training"), str)


def test_a_refusal_on_the_run_page_of_a_rebuilt_page_says_what_the_page_says(tmp_path):
    """A failure with a refusal after something was spent stands on the run's page, in the words of the page it stands on;
    an old page keeps the table's own."""
    from core.refusals import Refusal
    app = qt_app()
    window, page, _clock = _a_page(tmp_path)
    stand_in = StandIn()
    _start(page, stand_in)
    stand_in.solver(3)
    stand_in.fail(Refusal("The number of simulation batches must be at least 1; got 0.", field="num_runs"))
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.run_view.failure_message.text().endswith("Set it in 'Batches' on this page."), (
        page.run_view.failure_message.text())
    page.run_view.back_button.click()
    panel = window.part("reduction")
    other = StandIn(word="Reduction")
    start_run(panel, other)
    other.solver(3)
    other.fail(Refusal("The number of simulation batches must be at least 1; got 0.", field="num_runs"))
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert panel.run_view.failure_message.text().endswith(
        "Set it in the 'Batches' box in step 3, Training, or step 6, Narrowing round.")
    panel.run_view.back_button.click()


# ── the fields ──
def test_the_process_state_reads_what_the_fonts_add():
    """What the bundled files add to the process cannot be taken back, and it is read as the rest of what a start sets is, so
    that a test that added more, or took this away, is seen; the reading itself registers them, once, so that the first
    test to ask for a font is told from no other."""
    from tests._fixtures import process_state
    state = process_state(qt_app())
    assert state["fonts with the toolkit"] is True
    assert list(state["font substitutions"].values()) == [(design.LETTERING.lower(),)]


def test_a_pages_clearing_leaves_the_commas_own_under_a_row():
    """The comma's sentence is the field's: a page's live check that finds nothing to say leaves it under the row while the
    comma stands; and with the comma gone it goes, a page's sentence set meanwhile staying a page's. The field itself takes
    an error back whatever it was, as every page of the window that is no form asks of it."""
    form = _the_form()
    row = form.add("learning_rate", FloatField("1,5"))
    row.widget.editingFinished.emit()
    assert row.line.label.text() == COMMA_SENTENCE and row.in_error()
    row.set_line("err", "Learning rate must be above 0.")
    assert row.line.label.text() == row.widget.error() == "Learning rate must be above 0."
    row.set_line("", "")
    assert row.line.label.text() == row.widget.error() == COMMA_SENTENCE, "a page's clearing took the comma's sentence too"
    row.set_line("err", "Learning rate must be above 0.")
    row.widget.setText("1.5")
    assert row.widget.error() == "Learning rate must be above 0.", "the comma took a page's sentence with it"
    row.set_line("", "")
    assert row.widget.error() == "" and row.line.isHidden()


def test_a_rule_that_says_its_sentence_under_a_field_as_it_draws_lets_it_go_when_the_field_is_mended():
    """A page's facts are worked out at every edit, and say their sentence under the field as they go: that is no edit, and
    the sentence read back is no reading of the field, or the page would loop, and a mended field stay in error."""
    form = _the_form()
    row = form.add("num_runs", IntField(None))
    heard = []

    def live_check():
        heard.append(1)
        if len(heard) > 20:
            raise AssertionError("saying a line under a field was heard as an edit of it")
        words = row.reading()
        row.set_line("err" if words else "", words or "")

    form.edited.connect(live_check)
    row.widget.setText("")
    live_check()
    assert row.line.label.text() == "Batches is blank."
    row.widget.setText("6")
    assert row.line.isHidden() and row.widget.error() == "" and not row.in_error()


@pytest.mark.display
def test_the_training_buttons_stand_inside_the_smallest_window_with_real_metrics(tmp_path):
    """Needs QT_QPA_PLATFORM=windows (the root conftest defaults to offscreen, where this is skipped). With the real
    platform's fonts and their metrics, at the least size the window takes on the screen it stands on, at the ordinary
    text size and at one and a half times it, the Training page's main button and the words beside it stand inside the
    window with nothing drawn over them: a prior in use, no name typed, the fold open and every fact said. The window is
    laid out where no screen shows it and is never shown or brought to the front: the window in front stays in front."""
    import ctypes
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QGuiApplication
    from core.gui import fonts
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, prior_stub
    app = QApplication.instance() or QApplication([])
    if QGuiApplication.platformName() != "windows":
        pytest.skip(f"needs the windows platform, got {QGuiApplication.platformName()!r}")
    user32 = ctypes.WinDLL("user32")
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    front = user32.GetForegroundWindow()
    for factor in (1.0, 1.5):
        with text_size(factor), _the_look(False):
            fonts.load_app_font(app)
            controller_on(tmp_path, Clock(at(8, 12, 4)))
            window = build_window()
            window.setAttribute(Qt.WA_DontShowOnScreen)
            available = window.screen().availableGeometry()
            _opening, least = design.window_sizes((available.width(), available.height()))
            window.setMinimumSize(1, 1)
            window.resize(*least)
            window.show()
            window.go_to("inference", "model")
            screen = window.part("inference")
            screen.config_panel.foot.main_button.click()
            screen.session = SbiSession(draft=screen.session.draft, cfg=_nad_cfg(), inf_prior=prior_stub())
            screen.refresh_gates()
            window.go_to("inference", "training")
            page = screen.posterior_panel
            page.form.fold.set_open(True)
            page.read_facts()                        # no name typed: the longest words beside the button stand
            for _ in range(5):
                app.processEvents()
            inside = QRect(0, 0, window.width(), window.height())
            for part in (page.foot.main_button, page.foot.reason):
                edge = QRect(part.mapTo(window, QPoint(0, 0)), part.size())
                assert part.isVisible() and inside.contains(edge), (factor, type(part).__name__, edge, inside)
                for x, y in ((edge.left() + 2, edge.center().y()), (edge.center().x(), edge.center().y()),
                             (edge.right() - 2, edge.center().y())):
                    found = window.childAt(QPoint(x, y))
                    assert found is part or (found is not None and part.isAncestorOf(found)), (
                        factor, type(part).__name__, "something stands over it", x, y, found)
            assert not user32.IsWindowVisible(int(window.winId())), "the window was shown on the screen"
            window.hide()
            app.processEvents()
    assert user32.GetForegroundWindow() == front, "the front was taken from the window the person works in"
