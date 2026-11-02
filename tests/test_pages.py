"""The pieces every rebuilt page is made of: a fact and its order, the words beside the main button, where a value came
from, the form with its rows, its fold and its name and note, the foot, "In use", and the page that holds them, with
when it reads and how it hosts a run. They are read here on a stand-in page as awkward as a real one
(``tests/_fixtures.build_step_page``), in a whole window where a page's place in it matters.
"""
import itertools

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from core.gui import design, fields
from core.gui.run.controller import LIVE_TEXT, NOT_READ_YET_TEXT
from core.gui.step_facts import LEADS, Fact, about_words, ordered, origin_words, reading_words, reason_beside
from core.gui.widgets.choice import Choice
from core.gui.widgets.form import FOLD_NOT_USUAL, FOLD_WORDS, NAME_FREE_TEXT, Form, NameAndNote, TickedField
from core.gui.widgets.labeled_inputs import COMMA_SENTENCE, FloatField, IntField
from tests._fixtures import build_part, qt_app
from tests._look_helpers import _the_look, text_size
from tests._runs import at

BLOCKS = "the name cannot be used"


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
    the digits; a box that stands for "No limit" is the usual value of a row whose usual value is none."""
    form, rate, _size, limit = _a_fold()
    assert (limit.at_usual(), form.fold.button.text()) == (True, "Rarely changed (3 settings)")
    for typed, usual in (("5e-4", True), ("0.00050", True), ("", False), ("0,0005", False), ("0.0006", False)):
        rate.widget.setText(typed)
        assert rate.at_usual() is usual, typed
    rate.widget.setText("0.0005")
    limit.widget.ticked_box.setChecked(False)
    limit.widget.field.setText("300")
    assert not limit.at_usual() and form.fold.button.text().endswith("1 not usual)")
    whole = _the_form()
    assert whole.add("chi_max_cycles", FloatField(20.0), usual=20).at_usual(), "20.0 is 20"


def test_the_folds_sentence_names_those_not_usual_first():
    form, rate, _size, limit = _a_fold()
    assert form.fold.summary.text() == ("Learning rate 0.0005; Max rows per batch (0 = auto) Automatic; "
                                        "Epoch limit No limit")
    limit.widget.ticked_box.setChecked(False)
    limit.widget.field.setText("300")
    assert form.fold.summary.text() == ("Epoch limit 300 epochs; Learning rate 0.0005; "
                                        "Max rows per batch (0 = auto) Automatic")
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
    alone.fold.judge()
    assert not alone.fold.isHidden() and alone.fold.button.text() == "Rarely changed (0 settings)"


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
    """The store's three causes, each in its own sentence, through the real call."""
    from core.artifacts import ArtifactStore
    qt_app()
    names = build_part(NameAndNote, kind="prior", required=True, store=ArtifactStore(tmp_path))
    for typed, words in (("a b", "bad artifact name"), ("20261001T222935", "shaped like an artifact id")):
        names.name.setText(typed)
        names.check_now()
        assert words in names.name_fact().sentence and names.name_fact().blocks == BLOCKS


def test_a_required_name_left_blank_blocks_with_its_own_words():
    names, store = _names()
    fact = names.name_fact()
    assert (fact.kind, fact.sentence, fact.blocks) == ("err", "Type a name.", "a name is needed")
    assert names.name.property("state") in (None, ""), "a blank name is no error to outline"
    assert store.asked == []
    optional, _store = _names(required=False)
    assert optional.name_fact() is None


def test_a_note_is_judged_at_every_keystroke_and_reads_no_disk():
    """Two hundred characters pass, 201 do not, and a second line does not; each says so at once, and nothing is read."""
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
