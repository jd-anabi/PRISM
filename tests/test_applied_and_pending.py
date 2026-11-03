"""What a step holds against what its page shows: "In use" and "Changed, not applied", the rail and Home that follow
them, the question asked before what a session holds is let go, the one table of a page's rules, and what is remembered
of what was applied. Read on the first step, Model and options, on sessions of stand-ins and of the window's own; and
the scan that keeps the shared question the only one a rebuilt page asks."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.gui import chain, record_words
from core.gui.session import SbiSession
from core.gui.step_facts import NOT_HELD, SelectionRow, units_words
from tests._dialogs import all_asked, asked, choose, forget_asked
from tests._fixtures import build_inference, build_window, qt_app

#: The pages of Parameter inference built on the step page, by their files beside it.
REBUILT = ("config_tab.py", "prior_tab.py")


def _first_step(session=None):
    """The inference screen, holding ``session`` when one is handed, and its first step."""
    qt_app()
    screen = build_inference(session=session)
    if session is not None:
        screen.refresh_gates()
    return screen, screen.config_panel


def _shown_first_step(tmp_path):
    """The first step in a whole window in sight, off the screen, on the test's own clock: (window, screen, page). The
    application's font is left as it is: a font set here would stand for every later test of the process."""
    from core.gui import design
    from tests._look_helpers import _settle
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import shown_off_screen
    qt_app()
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    window = build_window()
    design.set_room((3400, 1380))
    window.setMinimumSize(1, 1)
    shown_off_screen(window, 1300, 820)
    window.go_to("inference", "model")
    _settle()
    screen = window.part("inference")
    return window, screen, screen.config_panel


def _pick(choice, value) -> None:
    """Choose the item of ``choice`` that holds ``value``."""
    found = choice.findData(value)
    assert found >= 0, value
    choice.setCurrentIndex(found)


def _apply(page) -> None:
    """The main button, pressed: the reading that counts, then the apply."""
    assert page.foot.main_button.isEnabled(), page.foot.reason.text()
    page.foot.main_button.click()


def _line(page, key) -> tuple:
    """What the line under the row ``key`` says: its marker and its words, or two empty strings."""
    line = page.form.rows[key].line
    return (line.kind, line.words) if line.says() else ("", "")


def _remember(**values) -> None:
    """Values in the settings file, as a launch before this one left them."""
    from core.gui import settings
    qs = settings.settings()
    for key, value in values.items():
        qs.setValue(key, value)
    qs.sync()


# ── a row that picks what a step holds ──
def test_a_selection_row_reads_changed_only_against_what_is_held():
    """``changed`` compares what the row shows with what the step holds, never while it holds nothing; a value no
    comparison can be made with is no change; the line names what is in use, in the row's own words."""
    shown, held = [5], [NOT_HELD]
    row = SelectionRow("num", lambda: held[0], lambda: shown[0], lambda value: f"{value} things")
    assert row.changed() is False
    held[0] = 5
    assert row.changed() is False
    held[0] = 6
    assert row.changed() is True and row.line() == "Changed, not applied. In use: 6 things."

    class Spiteful:
        def __ne__(self, other):
            raise ValueError("no comparison")

    held[0] = Spiteful()
    assert row.changed() is False
    assert SelectionRow("k", lambda: 3, lambda: 3).line() == "Changed, not applied. In use: 3."


def test_units_are_named_in_the_order_of_the_alphabet():
    assert units_words(("pN", "ms", "nm", "kHz")) == "kHz ms nm pN"
    assert units_words(["MHz", "kHz", "s"]) == "kHz MHz s"
    assert units_words(()) == ""


# ── "Changed, not applied" ──
def test_a_changed_choice_reads_changed_not_applied_and_names_what_is_in_use():
    """Applied, Nadrowski with the frequency response: the kind changed back reads under its field "Changed, not applied.
    In use: Frequency response.", "In use" counts it, and the rail says it; a second choice changed counts two; each put
    back, all of it goes. Applied with the other kind, choosing the frequency response changes one choice: its three
    settings, held by no draft of that kind, are no change."""
    screen, page = _first_step()
    _pick(page.kind_choice, True)
    _apply(page)
    assert screen.session.draft.chi_mode is True and screen.session.draft.model == "NADROWSKI"
    assert _line(page, "chi_mode") == ("", "") and page.in_use.pending.isHidden()
    _pick(page.kind_choice, False)
    assert _line(page, "chi_mode") == ("warn", "Changed, not applied. In use: Frequency response.")
    assert page.in_use.pending.sentence.plain_text() == "Attention. 1 choice below is changed and not applied."
    state = screen.step_states()[0]
    assert (state.rail_words, state.pending) == ("Changed, not applied", True) and screen.pending() == {"model"}
    _pick(page.rot_choice, False)
    assert _line(page, "reparam_rotate") == ("warn", "Changed, not applied. In use: On.")
    assert page.in_use.pending.sentence.plain_text() == "Attention. 2 choices below are changed and not applied."
    _pick(page.rot_choice, True)
    _pick(page.kind_choice, True)
    assert _line(page, "chi_mode") == _line(page, "reparam_rotate") == ("", "")
    assert page.in_use.pending.isHidden() and screen.pending() == frozenset()
    state = screen.step_states()[0]
    assert (state.rail_words, state.pending) == ("Done: Nadrowski", False)
    _pick(page.kind_choice, False)
    _apply(page)
    _pick(page.kind_choice, True)
    assert [row.key for row in page.selection_rows() if row.changed()] == ["chi_mode"]


@pytest.mark.parametrize("key, change, words", [
    ("model", lambda page: _pick(page.model_combo, "HOPF"), "Nadrowski"),
    ("chi_n_freqs", lambda page: page.form.rows["chi_n_freqs"].widget.setText("5"), "6"),
    ("chi_k_pad", lambda page: page.form.rows["chi_k_pad"].widget.setText("11"), "12"),
    ("chi_max_cycles", lambda page: page.form.rows["chi_max_cycles"].widget.setText("12.5"), "20 drive cycles"),
    ("units", lambda page: _pick(page.units_choice, "typed"), "kHz ms nm pN, from the model's file"),
], ids=["model", "driven recordings", "slots", "lock-in limit", "units"])
def test_each_row_names_what_is_in_use_in_its_own_words(key, change, words):
    """Every row of the step is a selection row, and the line under each says the value held, as that row would show
    it: a row read from another row's value, or a value said in another row's words, is seen here."""
    screen, page = _first_step()
    _pick(page.kind_choice, True)
    _apply(page)
    change(page)
    assert _line(page, key) == ("warn", f"Changed, not applied. In use: {words}.")
    others = [other for other in page.form.rows if other != key and _line(page, other) != ("", "")]
    assert others == [], others
    assert [row.key for row in page.selection_rows()] == ["model", "chi_mode", "chi_n_freqs", "chi_k_pad",
                                                          "chi_max_cycles", "reparam_rotate", "units"]


def test_the_units_held_are_said_as_the_model_held_declares_them():
    """The units held from the model's file are those of the model held, not of the model chosen since."""
    _screen, page = _first_step()
    _apply(page)
    _pick(page.model_combo, "BP")
    _pick(page.units_choice, "typed")
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: kHz ms nm pN, from the model's file.")


def test_the_driven_recordings_are_no_change_once_the_other_kind_is_chosen():
    """Held as the frequency response with one of its settings changed, the step chosen back to the other kind counts
    the kind alone: the settings of the frequency response, read only while it is chosen, are no change then."""
    _screen, page = _first_step()
    _pick(page.kind_choice, True)
    _apply(page)
    page.form.rows["chi_n_freqs"].widget.setText("5")
    assert [row.key for row in page.selection_rows() if row.changed()] == ["chi_n_freqs"]
    _pick(page.kind_choice, False)
    assert [row.key for row in page.selection_rows() if row.changed()] == ["chi_mode"]
    assert page.in_use.pending.sentence.plain_text() == "Attention. 1 choice below is changed and not applied."


def test_a_draft_that_holds_no_rotation_is_no_change_of_rotation():
    """A draft that holds no rotation, as a draft not made on this page may, leaves the rotation unchanged whichever is
    shown."""
    from core.gui.session import ConfigDraft
    screen, page = _first_step(SbiSession(draft=ConfigDraft(model="NADROWSKI", labels=[], state_dep_drift=False)))
    for shown in (False, True):
        _pick(page.rot_choice, shown)
        assert screen.pending() == frozenset() and _line(page, "reparam_rotate") == ("", ""), shown


def test_under_units_stands_the_line_of_the_file_or_the_box_and_its_note():
    """From the model's file, the line that names the file stands under "Units"; typed here, the box and what typing
    them means stand in its place."""
    _screen, page = _first_step()
    page.form.fold.set_open(True)

    def standing():
        return [part.isVisibleTo(page) for part in (page.units_line, page.units_text, page.typed_note)]

    assert standing() == [True, False, False]
    _pick(page.units_choice, "typed")
    assert standing() == [False, True, True]
    _pick(page.units_choice, "file")
    assert standing() == [True, False, False]


def test_typed_units_are_changed_until_applied_and_read_in_their_words():
    """Typed units held are said in the order of the alphabet, with where they came from; the same words in another
    order make another draft, so they read changed."""
    screen, page = _first_step()
    _pick(page.units_choice, "typed")
    page.units_text.setText("nm s pN Hz")
    _apply(page)
    assert screen.session.draft.units_override == ("nm", "s", "pN", "Hz")
    assert page.in_use.sentence.plain_text().endswith("Units Hz nm pN s, typed here.")
    assert _line(page, "units") == ("", "")
    page.units_text.setText("Hz nm pN s")
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: Hz nm pN s, typed here.")
    _pick(page.units_choice, "file")
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: Hz nm pN s, typed here.")


def test_the_fold_counts_the_choices_changed_in_it_and_stays_as_it_was_left():
    """A choice of the fold that differs from what is held is counted on the fold's button, beside the settings not
    usual, so that a shut fold says what "In use" counts below it; a choice changed among the main settings is not. The
    fold opens for no change: shut, it stays shut, and opened, it stays open."""
    _screen, page = _first_step()
    fold, rows = page.form.fold, page.form.rows
    _pick(page.kind_choice, True)
    _pick(page.rot_choice, False)
    _apply(page)
    assert fold.button.text() == "Rarely changed (5 settings, 1 not usual)"
    rows["chi_n_freqs"].widget.setText("5")
    assert fold.button.text() == "Rarely changed (5 settings, 2 not usual, 1 changed)"
    _pick(page.rot_choice, True)
    rows["chi_n_freqs"].widget.setText("6")
    assert fold.button.text() == "Rarely changed (5 settings, 1 changed)" and not fold.is_open()
    _pick(page.model_combo, "HOPF")
    assert fold.button.text() == "Rarely changed (5 settings, 1 changed)"
    assert page.in_use.pending.sentence.plain_text() == "Attention. 2 choices below are changed and not applied."
    fold.set_open(True)
    _pick(page.rot_choice, False)
    assert fold.button.text() == "Rarely changed (5 settings, 1 not usual)" and fold.is_open()


def test_a_model_chosen_reads_its_units_file_again_and_a_drawing_does_not(monkeypatch):
    """The units file is read as its model is chosen, so a file written since says its new units; drawing the page,
    whatever it says of the units, reads it no more."""
    from core.Helpers import file_manager
    declared, reads = ["kHz ms nm pN"], []
    monkeypatch.setattr(file_manager, "parse_units_file", lambda path: (reads.append(path), declared[0].split())[1])
    _screen, page = _first_step()
    _pick(page.model_combo, "HOPF")
    _apply(page)
    read = len(reads)
    page.redraw()
    page.refresh_local_gates()
    page.read_facts()
    assert len(reads) == read, "a drawing read the units file again"
    declared[0] = "s pN nm ms"
    _pick(page.model_combo, "NADROWSKI")
    assert len(reads) == read + 1, "choosing a model read more than its own units file"
    page.refresh_local_gates()
    assert len(reads) == read + 1, "a drawing read the units file of the model held"
    _pick(page.model_combo, "HOPF")
    assert page.units_line.plain_text() == "ms nm pN s, from Units/hopf/units.txt"
    assert page.units_text.text() == "ms nm pN s"


def test_a_held_model_whose_units_file_cannot_be_read_is_named_by_its_file(monkeypatch):
    """Units held from a units file that gave no words are named by the file in words, never by nothing."""
    from core import cli
    from core.refusals import Refusal

    def missing(model):
        raise Refusal(f"Missing units file for model '{model}'", field="units")

    monkeypatch.setattr(cli, "resolve_units_file", missing)
    _screen, page = _first_step()
    _apply(page)
    _pick(page.units_choice, "typed")
    page.units_text.setText("nm ms pN kHz")
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: the model's units file.")


def test_nothing_reads_changed_while_nothing_is_in_use():
    """Before anything is applied every row may differ from the usual and none reads changed: there is nothing to differ
    from."""
    screen, page = _first_step()
    _pick(page.model_combo, "HOPF")
    _pick(page.kind_choice, True)
    page.form.rows["chi_n_freqs"].widget.setText("4")
    _pick(page.rot_choice, False)
    _pick(page.units_choice, "typed")
    assert [row.key for row in page.selection_rows() if row.changed()] == []
    assert all(_line(page, key)[0] != "warn" for key in page.form.rows) and page.in_use.pending.isHidden()
    assert screen.pending() == frozenset() and screen.step_states()[0].rail_words == "Now: choose a model"
    assert page.foot.main_button.isEnabled() and page.foot.reason.text() == ""


def test_the_apply_button_is_greyed_with_nothing_to_apply():
    """Once applied, with every choice as it is held, there is nothing to apply, and the button says so beside it; one
    choice changed gives it back; and the way on to Prior is offered."""
    screen, page = _first_step()
    assert page.next_button.isHidden()
    _apply(page)
    foot = page.foot
    assert not foot.main_button.isEnabled()
    assert foot.reason.text() == "Nothing to apply: these choices are in use." == foot.main_button.accessibleDescription()
    assert (page.next_button.text(), page.next_button.isHidden()) == ("Go to Prior", False)
    _pick(page.rot_choice, False)
    assert foot.main_button.isEnabled() and foot.reason.text() == ""


def test_a_model_inference_cannot_take_greys_apply_with_its_reason(monkeypatch):
    """A model of your own with a drive, or with no free parameter, cannot be inferred: the button is greyed with the
    reason, an "Error." stands first among the facts, and a press that reaches the page all the same applies nothing.
    The rail still reads that a model is to be chosen."""
    from core import registry
    screen, page = _first_step()
    for forced, words in ((True, "This model has external forcing, so parameter inference cannot use it."),
                          (False, "This model has no free parameters to infer, so parameter inference cannot use it.")):
        monkeypatch.setattr(registry, "is_user_model", lambda name: name == "HOPF")
        monkeypatch.setattr(registry, "is_sbi_user_model", lambda name: False)
        monkeypatch.setattr(registry, "user_model_has_forcing", lambda name, forced=forced: forced)
        _pick(page.model_combo, "HOPF")
        page.redraw()
        foot = page.foot
        assert not foot.main_button.isEnabled()
        assert foot.reason.text() == "Cannot start yet: this model cannot be used."
        first = foot.facts.rows()[0]
        assert (first.kind, first.sentence.plain_text()) == ("err", f"Error. {words}")
        page._build_config()
        assert screen.session.draft is None and all_asked() == []
        assert screen.step_states()[0].rail_words == "Now: choose a model"
        _pick(page.model_combo, "NADROWSKI")
        assert foot.main_button.isEnabled() and foot.facts.rows() == []


# ── the question before records are let go ──
def test_the_keep_answer_changes_nothing_and_the_other_answer_replaces_the_session(monkeypatch):
    """With records held, applying asks, with "Keep this session" as the answer Enter and Escape give, naming each
    record. Kept, the session is the very object it was; the other answer starts a new one that holds the new draft and
    nothing else. With nothing held it asks nothing."""
    screen, page = _first_step()
    forget_asked()
    _apply(page)
    assert all_asked() == [], "a session holding nothing was asked about"
    held = SbiSession(draft=screen.session.draft, cfg=object(),
                      inf_prior=SimpleNamespace(kind="prior", name="retrain_prior"),
                      posterior=SimpleNamespace(kind="posterior", name="retrain"))
    screen.session = held
    screen.refresh_gates()
    _pick(page.rot_choice, False)
    forget_asked()
    _apply(page)
    question = asked()
    assert (question.title, question.default, question.safe) == ("Start a new session?", "Keep this session",
                                                                 "Keep this session")
    assert question.answers == ("Apply and start a new session", "Keep this session") and not question.loses_work
    assert question.sentence == ("Applying releases what this session holds: the prior retrain_prior; the posterior "
                                 "retrain. A different model, units or kind of observation makes them invalid. Each "
                                 "stays in Records and can be loaded again.")
    assert screen.session is held and held.draft.reparam_rotate is True and held.inf_prior is not None
    choose(monkeypatch, default=True)
    _apply(page)
    assert screen.session is held
    choose(monkeypatch, words="Apply and start a new session")
    _apply(page)
    assert screen.session is not held and screen.session.draft.reparam_rotate is False
    assert (screen.session.cfg, screen.session.inf_prior, screen.session.posterior) == (None, None, None)


def test_session_contents_names_each_record_and_never_raises():
    """One phrase for each record held, in the steps' order, by the label the rail gives it; no configuration, which is
    never held without its prior; and nothing raises on a stand-in that says nothing."""
    screen, _page = _first_step()
    assert screen.session_contents() == []
    screen.session = SbiSession(draft=object(), cfg=object())
    assert screen.session_contents() == [], "a draft and a configuration are no records"
    screen.session = SbiSession(cfg=object(), inf_prior=object(), posterior=object(), observation=object())
    assert screen.session_contents() == ["the unnamed record"] * 3
    made = SimpleNamespace(created="2026-10-01T22:29:00")
    screen.session = SbiSession(inf_prior=SimpleNamespace(kind="prior", name="retrain_prior"),
                                posterior=SimpleNamespace(kind="posterior", name="", manifest=made),
                                observation=SimpleNamespace(kind="observation", name="cell_a"))
    phrases = screen.session_contents()
    assert phrases[0] == "the prior retrain_prior" and phrases[2] == "the observation cell_a"
    assert phrases[1].startswith("the unnamed posterior, 1 Oct 2026, "), phrases[1]
    assert not any("config" in phrase for phrase in phrases)
    from core.gui.screens import inference_screen
    assert not hasattr(inference_screen, "_held_ref")


# ── the chain ──
def test_a_step_changed_and_not_applied_is_not_the_step_in_hand():
    """A step that holds something with a choice changed is passed over for the step in hand, so Home's next step reads
    on from what is held; a step that holds nothing is never so passed over."""
    screen, page = _first_step()
    _apply(page)
    _pick(page.kind_choice, True)
    states = screen.step_states()
    assert states[0].pending and chain.step_in_hand(states) == "prior"
    assert chain.next_step(screen.session, states).heading == "Next: build or load a prior"
    assert [state.marker for state in states[:2]] == ["warn", "now"]
    empty = chain.step_states(SimpleNamespace(), pending=frozenset({"model"}))
    assert not empty[0].pending and chain.step_in_hand(empty) == "model"


def test_a_choice_changed_tells_the_rail_once_and_a_redraw_tells_it_nothing():
    """The rail follows an edit that makes a choice differ from what is held, or no longer differ: once each way. An
    edit that leaves it as it was, and a drawing of the page, work the states out no further."""
    screen, page = _first_step()
    _apply(page)
    heard = []
    screen.states_changed.connect(lambda states: heard.append(states[0].rail_words))
    _pick(page.kind_choice, True)
    _pick(page.rot_choice, False)
    page.redraw()
    page.refresh_local_gates()
    _pick(page.rot_choice, True)
    _pick(page.kind_choice, False)
    assert heard == ["Changed, not applied", "Done: Nadrowski"], heard


def test_a_page_on_a_session_of_stand_ins_reads_unchanged_and_raises_nothing():
    """Bare objects in every slot, and a draft that says only its model: nothing reads changed, the states are the
    chain's own for that session, and nothing the page draws raises."""
    for draft in (object(), SimpleNamespace(model="NADROWSKI")):
        session = SbiSession(draft=draft, cfg=object(), inf_prior=object(), posterior=object(), observation=object())
        screen, page = _first_step(session)
        assert screen.pending() == frozenset() and not any(row.changed() for row in page.selection_rows())
        page.refresh_local_gates()
        page.read_facts()
        assert screen.step_states() == chain.step_states(
            session, observation_on_offer=screen.tsnpe_panel.obs_picker.has_entries(), label=record_words.label_of)
        assert page.in_use_words()[0] in ("unnamed", "Nadrowski")


# ── the rules ──
#: Each rule of the first step at both of its ends: what is typed, the row the line stands under and the line it says,
#: or None where that end passes. A rule that reads one row's value for another's, or moves an end by one, is seen
#: here.
_RULE_ENDS = [
    ({"chi_n_freqs": "1"}, "chi_n_freqs", "Driven recordings per observation must be at least 2."),
    ({"chi_n_freqs": "2"}, None, None),
    ({"chi_n_freqs": "25", "chi_k_pad": "24"}, "chi_n_freqs", "Driven recordings per observation must be at most 24."),
    ({"chi_n_freqs": "24", "chi_k_pad": "24"}, None, None),
    ({"chi_k_pad": "1", "chi_n_freqs": "2"}, "chi_k_pad", "Slots for driven recordings must be at least 2."),
    ({"chi_k_pad": "2", "chi_n_freqs": "2"}, None, None),
    ({"chi_k_pad": "25"}, "chi_k_pad", "Slots for driven recordings must be at most 24."),
    ({"chi_n_freqs": "13"}, "chi_n_freqs",
     "Driven recordings per observation must be at most the slots for driven recordings, 12."),
    ({"chi_n_freqs": "12"}, None, None),
    ({"chi_max_cycles": "2"}, "chi_max_cycles", "Lock-in limit must be greater than 2."),
    ({"chi_max_cycles": "2.001"}, None, None),
    ({"chi_max_cycles": "1e999"}, "chi_max_cycles", "Lock-in limit must be a finite number."),
    ({"chi_n_freqs": ""}, "chi_n_freqs", "Driven recordings per observation is blank."),
    ({"chi_max_cycles": "20,5"}, "chi_max_cycles", "Use a point as the decimal sign."),
    ({"units": ""}, "units", "Units typed here are blank."),
    ({"units": "nm notaunit"}, "units", "Units typed here cannot be used: "),
    ({"units": "nm ms pN kHz"}, None, None),
]


@pytest.mark.parametrize("typed, key, line", _RULE_ENDS,
                         ids=[" ".join(f"{name}={text}" for name, text in typed.items()) for typed, _k, _l in _RULE_ENDS])
def test_each_rule_says_its_line_greys_apply_and_refuses_the_press_alike(typed, key, line):
    """The one table of the first step's rules is read as the fields are typed and at the press. A rule broken says its
    line under its field, built from the row's label, stands as an "Error." that greys the button, and is the press's
    refusal: the same words, with what was found and the usual value. Its other end says nothing and applies."""
    from core.refusals import Refusal
    screen, page = _first_step()
    _pick(page.kind_choice, True)
    for name, text in typed.items():
        if name == "units":
            _pick(page.units_choice, "typed")
            page.units_text.setText(text)
        else:
            page.form.rows[name].widget.setText(text)
    foot, refused = page.foot, []
    page._refusal = refused.append
    if key is None:
        assert [name for name, row in page.form.rows.items() if row.in_error()] == []
        assert foot.main_button.isEnabled() and foot.reason.text() == ""
        page._build_config()
        assert refused == [] and screen.session.draft is not None
        return
    row = page.form.rows[key]
    assert row.in_error() and row.line.words.startswith(line), (row.line.words, line)
    assert [name for name, other in page.form.rows.items() if other.in_error()] == [key]
    assert not foot.main_button.isEnabled() and foot.reason.text() == "Cannot start yet: a setting cannot be used."
    assert [(fact.kind, fact.sentence.plain_text()) for fact in foot.facts.rows()] == [("err", f"Error. {row.line.words}")]
    page._build_config()
    assert screen.session.draft is None and len(refused) == 1 and isinstance(refused[0], Refusal)
    assert refused[0].field == key and refused[0].message.startswith(line.rstrip(".")), refused[0].message


def test_a_rule_broken_in_a_field_off_the_page_stops_nothing():
    """The frequency response's three settings are read only while it is chosen: what they hold under the other kind is
    no fault, and applies as nothing."""
    screen, page = _first_step()
    _pick(page.kind_choice, True)
    page.form.rows["chi_n_freqs"].widget.setText("1")
    assert not page.foot.main_button.isEnabled()
    _pick(page.kind_choice, False)
    assert page.foot.main_button.isEnabled() and page.problems() == {}
    _apply(page)
    assert screen.session.draft.chi_n_freqs is None


def test_a_change_of_kind_judges_the_driven_settings_at_once():
    """A comma left in a setting of the frequency response while the other kind was chosen greys the button the moment
    the frequency response is chosen, with no reading after it: the settings stand before the page judges them."""
    _screen, page = _first_step()
    page.form.rows["chi_max_cycles"].widget.setText("20,5")
    assert page.foot.main_button.isEnabled()
    _pick(page.kind_choice, True)
    assert _line(page, "chi_max_cycles") == ("err", "Use a point as the decimal sign.")
    assert not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: a setting cannot be used."


def test_typed_units_are_judged_as_they_are_typed(tmp_path):
    """The units typed under the row are heard as the row's own: a word that is no unit says so under the row as it is
    typed, a mended one takes it away, and a refusal of the units stands only until they are typed again."""
    from core.refusals import Refusal
    screen, page = _first_step()
    _pick(page.units_choice, "typed")
    page.units_text.setText("nm ms pN kHzz")
    assert _line(page, "units")[0] == "err" and not page.foot.main_button.isEnabled()
    page.units_text.setText("nm ms pN kHz")
    assert _line(page, "units") == ("", "") and page.foot.main_button.isEnabled()
    page._refusal(Refusal("The units were refused.", field="units"))
    assert _line(page, "units") == ("err", "The units were refused.")
    assert page.units_text.property("state") == "error", "a refusal of the units left their box without its outline"
    page.units_text.setText("nm ms pN Hz")
    assert _line(page, "units") == ("", "") and page.units_text.property("state") == ""


def test_typed_units_in_error_outline_their_box():
    """The box of typed units is outlined while what it holds is in error, as every field in error is, and no longer
    once it is mended: a box a row hears as its own field takes the row's outline. A line under the row that is no
    error, "Changed, not applied", outlines nothing."""
    _screen, page = _first_step()
    _apply(page)
    _pick(page.units_choice, "typed")
    assert _line(page, "units")[0] == "warn" and page.units_text.property("state") == ""
    page.units_text.setText("nm ms pN kHzz")
    assert page.units_text.property("state") == "error"
    page.units_text.setText("nm ms pN kHz")
    assert _line(page, "units")[0] == "warn" and page.units_text.property("state") == ""


def test_units_filled_from_a_file_follow_the_model_until_they_are_typed():
    """Typed here, the box still holds what the model's units file filled it with until a person types: a model chosen
    then fills it from its own file, so that the units of the model before it are never applied as typed for this one.
    Once typed, no model chosen writes over them."""
    _screen, page = _first_step()
    _pick(page.units_choice, "typed")
    assert page.units_text.text() == "kHz ms nm pN"
    _pick(page.model_combo, "BP")
    assert page.units_text.text() == "ms nm"
    page.units_text.setText("nm")
    _pick(page.model_combo, "HOPF")
    assert page.units_text.text() == "nm"


def test_a_refusal_whose_cure_is_in_another_field_stops_the_press_only_until_the_page_reads_again(tmp_path):
    """A refusal marks its row and stops the press. An edit of another field mends nothing of the row's own, and still
    the button comes back once the page reads again, after the typing has stopped, though the first step reads nothing
    saved; and the press asks the rules once more."""
    from core.refusals import Refusal
    window, screen, page = _shown_first_step(tmp_path)
    _pick(page.kind_choice, True)
    page._refusal(Refusal("The slots are refused.", field="chi_k_pad"))
    row = page.form.rows["chi_k_pad"]
    assert (row.line.kind, row.line.words) == ("err", "The slots are refused.")
    assert not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: the last start was refused."
    page.form.rows["chi_n_freqs"].widget.setText("5")
    assert not page.foot.main_button.isEnabled() and row.line.words == "The slots are refused."
    assert page.read_timer.isActive(), "an edit on a page that reads nothing saved started no reading"
    page.read_timer.timeout.emit()
    page.read_timer.stop()
    assert page.foot.main_button.isEnabled() and (row.line.kind, row.line.words) == ("warn", "The slots are refused.")
    _apply(page)
    assert screen.session.draft.chi_n_freqs == 5


def test_a_comma_in_the_lock_in_limit_greys_apply_with_its_sentence_under_the_field(tmp_path):
    """On the real page, with real keys: the comma is the field's own sentence, under the field and among the facts, and
    the button is greyed until it goes."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from tests._look_helpers import _settle
    from tests._window_helpers import in_front
    window, screen, page = _shown_first_step(tmp_path)
    _pick(page.kind_choice, True)
    page.form.fold.set_open(True)
    field = page.form.rows["chi_max_cycles"].widget
    with in_front(window):
        field.setFocus(Qt.TabFocusReason)
        field.selectAll()
        QTest.keyClicks(field, "20,5")
        QTest.keyClick(field, Qt.Key_Tab)
        _settle()
    assert _line(page, "chi_max_cycles") == ("err", "Use a point as the decimal sign.")
    assert not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: a setting cannot be used."
    field.setText("20.5")
    assert _line(page, "chi_max_cycles") == ("", "") and page.foot.main_button.isEnabled()


# ── what is remembered ──
def test_the_model_and_the_kind_come_back_kept_from_last_time_until_touched():
    """What was last applied is put back as a choice, each of the two rows saying it was kept from last time until it is
    touched, and nothing is applied: the step holds nothing. Applying keeps the word on what was not touched."""
    _remember(**{"inference_config/model": "HOPF", "inference_config/chi_mode": "true"})
    screen, page = _first_step()
    rows = page.form.rows
    assert (page.model_combo.currentText(), page.kind_choice.currentData()) == ("HOPF", True)
    assert [key for key, row in rows.items() if row.origin.text()] == ["model", "chi_mode", "chi_f0", "chi_freq_bounds"]
    assert rows["model"].origin.text() == rows["chi_mode"].origin.text() == "kept from last time"
    assert screen.session.draft is None and page.in_use.line_text() == "In use: nothing yet."
    assert page.in_use.sentence.plain_text() == "Choose the model and the kind of observation, then apply them."
    _pick(page.model_combo, "NADROWSKI")
    assert rows["model"].origin.text() == "" and rows["chi_mode"].origin.text() == "kept from last time"
    _apply(page)
    assert rows["chi_mode"].origin.text() == "kept from last time" and rows["model"].origin.text() == ""
    _pick(page.kind_choice, False)
    _pick(page.kind_choice, True)
    assert rows["chi_mode"].origin.text() == "", "a kind touched and put back is still said to be kept"


@pytest.mark.parametrize("written", ["false", "0"])
def test_a_kind_of_the_bounds_file_comes_back_kept_from_last_time(written):
    """The kind last applied as the bounds file's comes back however the settings file wrote it, and is said to be kept
    from last time."""
    _remember(**{"inference_config/chi_mode": written})
    _screen, page = _first_step()
    assert page.kind_choice.currentData() is False
    assert page.form.rows["chi_mode"].origin.text() == "kept from last time"


@pytest.mark.parametrize("model, kind", [("NOT_A_MODEL", "maybe"), ("", "")], ids=["stale", "blank"])
def test_what_cannot_be_put_back_opens_as_usual_and_is_not_said_to_be_kept(model, kind):
    _remember(**{"inference_config/model": model, "inference_config/chi_mode": kind})
    _screen, page = _first_step()
    rows = page.form.rows
    assert (page.model_combo.currentText(), page.kind_choice.currentData()) == ("NADROWSKI", False)
    assert rows["model"].origin.text() == rows["chi_mode"].origin.text() == ""


def test_a_remembered_model_taken_from_the_list_is_no_longer_said_to_be_kept(tmp_path):
    """A model of your own put back from last time and then deleted: the list falls to Nadrowski, which nobody kept."""
    from core import registry
    registry.register(registry.ModelSpec("UMTESTKEPT", ["a"], is_user_model=True, n_vars=1))
    try:
        _remember(**{"inference_config/model": "UMTESTKEPT"})
        window = build_window()
        page = window.part("inference").config_panel
        assert page.form.rows["model"].origin.text() == "kept from last time"
    finally:
        registry.unregister("UMTESTKEPT")
    window._on_user_models_changed()
    assert page.model_combo.currentText() == "NADROWSKI" and page.form.rows["model"].origin.text() == ""


def test_applying_remembers_the_model_and_the_kind_and_nothing_else(monkeypatch):
    """The two keys are written when the step is applied, by the one writer, together; a draft that is not the window's
    own writes nothing it could not read back; and the step writes nothing as the window closes."""
    from core.gui import remembered, settings
    written = []
    real = remembered.put_many
    monkeypatch.setattr(remembered, "put_many", lambda pairs: (written.append(list(pairs)), real(pairs))[1])
    screen, page = _first_step()
    _pick(page.model_combo, "HOPF")
    _pick(page.kind_choice, True)
    _pick(page.rot_choice, False)
    assert written == []
    _apply(page)
    assert written == [[("inference_config/model", "HOPF"), ("inference_config/chi_mode", True)]]
    qs = settings.settings()
    assert (qs.value("inference_config/model"), settings.get_bool(qs, "inference_config/chi_mode", False)) == (
        "HOPF", True)
    screen.new_draft(SimpleNamespace(model="BP", chi_mode="yes"))

    class NoName:
        """A model that is no string, as a draft that is not the window's own may hold: it says only its folder."""

        def lower(self):
            return "bp"

    screen.new_draft(SimpleNamespace(model=NoName(), chi_mode=True))
    assert written[1:] == [[("inference_config/model", "BP")], [("inference_config/chi_mode", True)]], written
    page.save_settings(qs)
    qs.sync()
    again = settings.settings()
    assert sorted(again.allKeys()) == ["inference_config/chi_mode", "inference_config/model"]
    assert settings.get_bool(again, "inference_config/chi_mode", False) is True


# ── the page ──
def test_applying_starts_no_run_and_its_page_shows_none(monkeypatch):
    """The first step applies at once, on the window's thread: no run is asked for, so no run's page ever takes the
    place of its form."""
    from core.gui.run.controller import RunController, controller
    started = []
    monkeypatch.setattr(RunController, "dispatch", lambda held, *args, **kwargs: started.append(args))
    screen, page = _first_step()
    _apply(page)
    assert started == [] and not controller().is_live() and page.results.currentWidget() is page.figures


def test_the_first_step_reads_nothing_saved_and_says_no_reading():
    """It holds nothing saved to read: no line of when the records were read, no "Check again"."""
    _screen, page = _first_step()
    page.read_facts()
    assert page.reads_the_disk is False and page.foot.reading.isHidden() and page.foot.check_again.isHidden()


def test_a_rebuilt_page_keeps_no_message_box():
    """A rebuilt page asks and tells through the one shared question: neither the step page nor any page built on it names
    the toolkit's message box, or keeps a log pane of its own."""
    folder = Path(__file__).resolve().parents[1] / "core" / "gui" / "panels" / "inference"
    for name in ("step_page.py", *REBUILT):
        tree = ast.parse((folder / name).read_text(encoding="utf-8"))
        named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        named |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert not {"QMessageBox", "log_pane"} & named, name


def test_a_model_of_your_own_says_its_checks_are_not_tuned(monkeypatch):
    """A model of your own that parameter inference can take is told, before it is applied, that the checks are not tuned
    for it, and where to read them; it blocks nothing. One inference cannot take says only why; a built-in model says
    nothing."""
    from core import registry
    _screen, page = _first_step()
    usable = [True]
    monkeypatch.setattr(registry, "is_user_model", lambda name: name == "HOPF")
    monkeypatch.setattr(registry, "is_sbi_user_model", lambda name: usable[0])
    monkeypatch.setattr(registry, "user_model_has_forcing", lambda name: True)
    _pick(page.model_combo, "HOPF")
    rows = page.foot.facts.rows()
    assert [(row.kind, row.sentence.plain_text()) for row in rows] == [
        ("info", "Information. The checks are not tuned for a model of your own: read their results in step 4, Checks, "
                 "before you rely on its posterior.")]
    assert page.foot.main_button.isEnabled() and page.foot.reason.text() == ""
    usable[0] = False
    page.redraw()
    assert [row.kind for row in page.foot.facts.rows()] == ["err"]
    _pick(page.model_combo, "NADROWSKI")
    assert page.foot.facts.rows() == []


@contextlib.contextmanager
def _first_step_with_the_ring(tmp_path, monkeypatch):
    """The first step in the whole program, whose ring follows the keyboard as it does for a person, in sight off the
    screen and in front, on the test's own clock: (window, screen, page)."""
    from tests._fixtures import whole_program
    from tests._look_helpers import _settle
    from tests._runs import Clock, at, controller_on
    from tests._window_helpers import in_front, shown_off_screen
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "model")
        _settle()
        screen = window.part("inference")
        with in_front(window):
            yield window, screen, screen.config_panel
        window.hide()


def test_only_a_press_by_the_keyboard_hands_it_on_to_the_next_step(tmp_path, monkeypatch):
    """The main button greys as it applies. Pressed by the keyboard, Enter or Space with the ring on it, the keyboard goes
    on to the way to the next step, ring and all, and not where the toolkit hands it from a greyed button; so it does with
    records held, the question answered, though the question held the keyboard while it stood. The access key presses
    the button without giving it the keyboard, which stays where it was. A press by the mouse gives the button the
    keyboard with no ring, and the page hands it nowhere: the toolkit gives it to the step's row of the rail. A press by
    the mouse on the button while it wears the ring counts as the keyboard's, and hands the keyboard on with no ring."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui import style
    from tests._look_helpers import _arrive, _settle
    from tests._runs import turn_until
    with _first_step_with_the_ring(tmp_path, monkeypatch) as (window, screen, page):
        button, follower = page.foot.main_button, style.current_follower()

        def ring_on(widget):
            widget.setFocus(Qt.TabFocusReason)
            _arrive(widget, Qt.TabFocusReason)
            assert follower.target() is widget, "the ring is not where the keyboard was brought"

        def one_more_change():
            _pick(page.rot_choice, not page.rot_choice.currentData())
            assert button.isEnabled()

        for key in (Qt.Key_Return, Qt.Key_Space):
            ring_on(button)
            QTest.keyClick(button, key)
            _settle()
            assert screen.session.draft.reparam_rotate == page.rot_choice.currentData(), key
            assert QApplication.focusWidget() is page.next_button and follower.target() is page.next_button, key
            one_more_change()
        ring_on(page.model_combo)
        QTest.keyClick(page.model_combo, Qt.Key_A, Qt.AltModifier)
        turn_until(QApplication.instance(), lambda: screen.session.draft.reparam_rotate == page.rot_choice.currentData())
        _settle()
        assert QApplication.focusWidget() is page.model_combo and follower.target() is page.model_combo
        one_more_change()
        ring_on(page.model_combo)
        QTest.mouseClick(button, Qt.LeftButton)
        _settle()
        assert screen.session.draft.reparam_rotate == page.rot_choice.currentData()
        assert QApplication.focusWidget() is window.shell.rail.row("model") and follower.target() is None
        one_more_change()
        ring_on(button)
        QTest.mouseClick(button, Qt.LeftButton)
        _settle()
        assert screen.session.draft.reparam_rotate == page.rot_choice.currentData()
        assert QApplication.focusWidget() is page.next_button and follower.target() is None
        one_more_change()
        screen.session = SbiSession(draft=screen.session.draft, cfg=object(),
                                    inf_prior=SimpleNamespace(kind="prior", name="retrain_prior"))
        screen.refresh_gates()
        ring_on(button)
        choose(monkeypatch, words="Apply and start a new session", during=button.clearFocus)
        QTest.keyClick(button, Qt.Key_Return)
        _settle()
        assert asked().title == "Start a new session?" and screen.session.inf_prior is None
        assert QApplication.focusWidget() is page.next_button and follower.target() is page.next_button


# ── the second step: a prior comes into use when its run returns ──
def _held_records():
    """A configuration, a prior, a posterior and an observation, as stand-ins that say what a page reads of each."""
    from core.config import BOUNDS_PATH
    cfg = SimpleNamespace(model="NADROWSKI", sources={"bounds": str(BOUNDS_PATH / "nadrowski" / "master.txt")},
                          params_dict={},
                          rescale_params={}, force_params_dict={}, observation_mode="spontaneous",
                          check_unit_consistency=lambda: [])
    prior = SimpleNamespace(kind="prior", id="p_old", name="old_prior",
                            manifest=SimpleNamespace(created="2026-10-01T22:29:00+00:00",
                                                     inputs={"bounds": {"path": "Bounds/nadrowski/master.txt"}},
                                                     config={"param_keys": ["k", "x_scale"]}))
    posterior = SimpleNamespace(kind="posterior", id="post_old", name="retrain")
    observation = SimpleNamespace(kind="observation", id="obs_old", name="cell_a")
    return cfg, prior, posterior, observation


@contextlib.contextmanager
def _second_step(tmp_path, monkeypatch, *, held=True):
    """The second step in a whole window in sight, off the screen, on the test's own clock, its store the test's own:
    the first step applied, Nadrowski with the bounds file's kind; with ``held``, a configuration, a prior, a posterior
    and an observation held; the master bounds file selected, the build shown, a name and a note typed, and the
    stage's build a stand-in run: (window, screen, page, the stand-in run)."""
    from core import orchestrator
    from core.artifacts import ArtifactStore, use_store
    from tests._runs import StandIn
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window, screen, first = _shown_first_step(tmp_path)
        _apply(first)
        if held:
            cfg, prior, posterior, observation = _held_records()
            screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior, posterior=posterior,
                                        observation=observation)
            screen.refresh_gates()
        window.go_to("inference", "prior")
        page = screen.prior_panel
        page.show_view("build")
        page.bounds_picker.restore_key("master.txt")
        page.names.name.setText("retrain_2")
        page.names.note.setText("a second try")
        page.read_facts()
        stand_in = StandIn(word="Prior build")
        monkeypatch.setattr(orchestrator, "build_prior", stand_in)
        yield window, screen, page, stand_in


def _press_and_end(page, stand_in, end, monkeypatch):
    """Press the build, and end its run as ``end`` names: refused at the stage's entry, failed, stopped, or taken back
    as it starts."""
    from core.gui.run.controller import RunController, controller
    from core.refusals import Refusal
    from tests._runs import turn_until
    app = qt_app()
    if end == "withdrawn":
        def broken(_self):
            raise TypeError("the first drawing failed")
        monkeypatch.setattr(RunController, "_draw", broken)
        with pytest.raises(TypeError):
            page._main_clicked()
        return
    page._main_clicked()
    assert controller().is_live() and stand_in.started.wait(5)
    if end == "refused before the spend":
        stand_in.fail(Refusal("A prior named 'retrain_2' was made meanwhile.", field="name"))
    elif end == "failed":
        stand_in.fail(RuntimeError("the sweep found no stable set"))
    else:
        controller().request_stop()
    turn_until(app, lambda: not controller().is_live(), 10.0)


@pytest.mark.parametrize("end", ["refused before the spend", "failed", "stopped", "withdrawn"])
def test_a_build_that_ends_any_way_but_well_leaves_what_was_held(end, tmp_path, monkeypatch):
    """A build that is refused at the stage's entry, fails, is stopped or is taken back as it starts writes no slot: the
    configuration, the prior, the posterior and the observation held are the very objects they were, the name and the
    note stand as typed, and the rail and "In use" say what they said."""
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        said = (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("prior").accessibleName())
        _press_and_end(page, stand_in, end, monkeypatch)
        assert screen.session is session
        assert all(now is then for now, then in zip((session.cfg, session.inf_prior, session.posterior,
                                                     session.observation), held)), end
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try"), end
        assert (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("prior").accessibleName()) == said, end


def test_a_prior_comes_into_use_only_when_its_run_returns(tmp_path, monkeypatch):
    """The press holds nothing new. The prior the run returns comes into use with the very configuration the press
    handed the run, while the run is still live: what depended on the prior held goes with it, the states are said once,
    on the window's thread, and a training continued with the session is one made with what it holds. At the run's end
    the saved list is read again and moved to the new prior, the name and the note are emptied, and the way to Training
    stands."""
    import threading
    from core.gui.run import retry
    from core.gui.run.controller import controller
    from core.gui.run.model import RunSpec
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        session = screen.session
        old_cfg, old_prior = session.cfg, session.inf_prior
        candidate = page._candidate
        assert candidate is not None and candidate is not old_cfg
        heard, threads = [], []
        screen.states_changed.connect(lambda states: (heard.append(states), threads.append(threading.current_thread())))
        page._main_clicked()
        assert stand_in.started.wait(5)
        args, kwargs = stand_in.calls[-1]
        assert args[0] is candidate and args[1:] == (None, True)
        assert session.cfg is old_cfg and session.inf_prior is old_prior
        assert not page.switch_button.isEnabled(), "the way to the other view stood open during the run"
        made = SimpleNamespace(kind="prior", id="p_new", name="retrain_2",
                               manifest=SimpleNamespace(created="2026-10-08T12:04:00+00:00",
                                                        inputs={"bounds": {"path": "Bounds/nadrowski/master.txt"}},
                                                        config={"param_keys": ["k"] * 13}))
        taken = []

        def take(cfg, loaded, kept=screen.take_prior):
            said = len(heard)
            kept(cfg, loaded)
            taken.append((controller().is_live(), threading.current_thread(), len(heard) - said))

        monkeypatch.setattr(screen, "take_prior", take)
        stand_in.release(made)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert taken == [(True, threading.main_thread(), 1)], "the prior was taken after its run, or more than once"
        assert screen.session is session and session.cfg is candidate and session.inf_prior is made
        assert session.posterior is None and session.observation is None
        assert set(threads) == {threading.main_thread()}
        training = RunSpec("Training", ("inference", "training")).with_call(lambda *a: None,
                                                                            (session.cfg, session.inf_prior), {})
        assert retry.made_in(training, session)
        assert page.in_use.line_text() == "In use: retrain_2"
        assert page.in_use.sentence.plain_text() == "Built on master.txt, 13 parameters. Saved as a record."
        assert (page.names.name.text(), page.names.note.text()) == ("", "")
        assert not page.next_button.isHidden() and page.next_button.text() == "Go to Training"
        assert page.switch_button.isEnabled()


@pytest.mark.parametrize("build", [True, False], ids=["a build", "a load"])
def test_another_prior_asks_before_it_releases_a_posterior(build, tmp_path, monkeypatch):
    """With a posterior held, another prior asks first, "Keep the posterior" preselected: kept, nothing starts; the other
    answer starts the run, and the posterior and the observation held go only when its prior comes back. The question
    names the observation with the posterior, or the posterior alone."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        if not build:
            page.show_view("saved")
            page.prior_picker.combo.addItem("old_prior", "p_old")
            page.prior_picker.combo.addItem("other_prior", "p_two")
            page.prior_picker.restore_key("p_two")
        page.main_pressed()
        question = asked()
        title, other = ("Build a new prior?", "Build and release the posterior") if build else (
            "Load this prior?", "Load and release the posterior")
        once = "is built" if build else "has loaded"
        assert (question.title, question.answers, question.default) == (title, (other, "Keep the posterior"),
                                                                        "Keep the posterior")
        assert question.sentence == (
            f"This session holds the posterior retrain and the observation cell_a. They belong to the prior in use, so "
            f"another prior releases them once it {once}. Each stays in Records and can be loaded again.")
        assert not controller().is_live() and stand_in.calls == []
        screen.session.observation = None
        forget_asked()
        page.main_pressed()
        assert asked().sentence == (f"This session holds the posterior retrain. It belongs to the prior in use, so "
                                    f"another prior releases it once it {once}. It stays in Records and can be loaded "
                                    "again.")
        choose(monkeypatch, words=other)
        page.main_pressed()
        assert controller().is_live() and stand_in.started.wait(5)
        assert screen.session.posterior is not None
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert screen.session.posterior is None


def test_a_prior_held_with_no_posterior_is_replaced_without_a_question(tmp_path, monkeypatch):
    """No posterior, no question: an observation held alone is let go by the new prior, unasked."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        screen.session.posterior = None
        forget_asked()
        page.main_pressed()
        assert all_asked() == [] and stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert screen.session.observation is None and screen.session.inf_prior.id == "p_new"


def test_the_bounds_file_changed_reads_changed_not_applied(tmp_path, monkeypatch):
    """With a prior held, another bounds file selected reads "Changed, not applied" under its list, naming the file in
    use, and the rail says so; the limits typed by hand chosen say it under their tick, in the fold, which counts it;
    limits typed by hand in use are said as such; put back, all of it goes. Saved, the prior chosen is the other row
    that picks what the step holds."""
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        assert _line(page, "bounds") == ("", "")
        page.typed_tick.setChecked(True)
        assert _line(page, "bounds") == ("", "")
        assert _line(page, "bounds_typed") == ("warn", "Changed, not applied. In use: master.txt.")
        assert page.form.fold.button.text() == "Rarely changed (8 settings, 1 not usual, 1 changed)"
        page.typed_tick.setChecked(False)
        assert _line(page, "bounds_typed") == ("", "") and screen.pending() == frozenset()
        page.bounds_picker.restore_key("master_tier1.txt")
        assert _line(page, "bounds") == ("warn", "Changed, not applied. In use: master.txt.")
        assert screen.pending() == {"prior"} and screen.step_states()[1].rail_words == "Changed, not applied"
        page.bounds_picker.restore_key("master.txt")
        assert _line(page, "bounds") == ("", "") and screen.pending() == frozenset()
        screen.session.cfg.sources = {}
        screen.session.cfg.params_dict = {"k": (None, (0.01, 5.0))}
        page.redraw()
        assert _line(page, "bounds") == ("warn", "Changed, not applied. In use: limits typed by hand.")
        page.show_view("saved")
        page.prior_picker.combo.addItem("old_prior", "p_old")
        page.prior_picker.restore_key("p_old")
        from core.config import BOUNDS_PATH
        screen.session.cfg.sources = {"bounds": str(BOUNDS_PATH / "nadrowski" / "master.txt")}
        page.redraw()
        assert _line(page, "prior") == ("", "") and page.nothing_to_apply() is True
        page.prior_picker.combo.addItem("other_prior", "p_two")
        page.prior_picker.restore_key("p_two")
        assert _line(page, "prior") == ("warn", "Changed, not applied. In use: old_prior.")


def test_a_load_that_fails_says_so_in_the_shared_question(tmp_path, monkeypatch):
    """A load is never handed the run's page, so its failure is told in the shared question, the error's type and its
    trace behind its button, and stands after it as the page's own line; nothing held changes. A refusal at the load is
    told as every refusal is."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Load and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        held = screen.session.inf_prior
        two = _prior_artifact(default_store(), _nad_cfg(), name="other_prior").id
        page.show_view("saved")
        page.prior_picker.refresh()
        page.prior_picker.restore_key(two)
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        forget_asked()
        stand_in.fail(OSError("prior.pt cannot be read"))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        error = asked()
        assert (error.kind, error.title, error.marker, error.answers) == ("error", "Error", "err", ("OK",))
        assert error.sentence == "prior.pt cannot be read"
        assert error.detail.startswith("OSError: prior.pt cannot be read") and "Traceback" in error.detail
        said = page.in_use.said
        assert not said.isHidden() and said.sentence.plain_text() == "Attention. prior.pt cannot be read"
        assert screen.session.inf_prior is held and page.results.currentWidget() is page.figures
        forget_asked()
        page.main_pressed()
        assert stand_in.started.wait(5)
        assert said.isHidden(), "the page's own line stood through its next run"
        stand_in.fail(Refusal("Prior 'other_prior' does not match this configuration.", field="prior"))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        assert [box.kind for box in all_asked()][-1] == "refusal"
        assert _line(page, "prior") == ("err", "Prior 'other_prior' does not match this configuration.")


def test_a_prior_saved_under_another_name_says_so(tmp_path, monkeypatch):
    """A record that comes back under another name than the one typed, taken while the run was going, says so under
    "In use"; the name and the note are emptied all the same, for the record was made."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        said = page.in_use.said
        assert not said.isHidden() and said.sentence.plain_text() == (
            "Attention. It was saved without its name: 'retrain_2' was taken while the run was going.")
        assert (page.names.name.text(), page.names.note.text()) == ("", "")


def test_the_stop_and_end_of_a_prior_build_say_what_is_true_of_the_page(tmp_path, monkeypatch):
    """The words a prior build's run says of its stop and its end hold on the page that starts it: its stop asks first and
    says nothing of the prior is kept, and the prior in use stays in use; a build stopped keeps its page until "Back to
    the form", and the form then holds the name and the note as typed; one finished gives the column back with "How the
    run went" over the new prior's figure, its alone."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        held = screen.session.inf_prior
        page.main_pressed()
        assert stand_in.started.wait(5)
        assert page.results.currentWidget() is page.run_view
        choose(monkeypatch, words="Stop the build")
        forget_asked()
        page.run_view.stop_button.click()
        question = asked()
        assert (question.title, question.answers, question.default) == (
            "Stop the prior build?", ("Stop the build", "Keep building"), "Keep building")
        assert question.sentence.endswith("Nothing of the prior is kept; it is built again from the start next time.")
        choose(monkeypatch, words="Build and release the posterior")
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert page.results.currentWidget() is page.run_view and screen.session.inf_prior is held
        page.run_view.back.emit()
        assert page.results.currentWidget() is page.figures
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try")
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.figure("Prior")
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        assert page.results.currentWidget() is page.figures and page.run_view.how_link.isVisibleTo(page.figures)
        assert [page.figure_stack.tabText(i) for i in range(page.figure_stack.count())] == ["Prior"]
        assert page.in_use.line_text() == "In use: retrain_2"


def _settle_events():
    from tests._look_helpers import _settle
    _settle()


def test_a_load_by_the_keyboard_hands_it_on_to_training_and_one_by_the_mouse_does_not(tmp_path, monkeypatch):
    """A load greys the main button while it runs, and no run's page takes the keyboard. Pressed by the keyboard, Enter
    with the ring on it, the keyboard goes on to the way to Training once the prior has come back, ring and all, and not
    where the toolkit put it from the greyed button; a load that fails gives it back to the main button once its question
    is answered. Pressed by the mouse, the page hands it nowhere: the toolkit gives it to the step's row of the rail."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core import orchestrator
    from core.artifacts import ArtifactStore
    from core.gui import style
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._look_helpers import _arrive, _settle
    from tests._runs import StandIn, turn_until
    stand_in = StandIn(word="Prior load")
    monkeypatch.setattr(orchestrator, "build_prior", stand_in)
    with _first_step_with_the_ring(tmp_path, monkeypatch) as (window, screen, first):
        _apply(first)
        window.go_to("inference", "prior")
        _settle()
        page = screen.prior_panel
        store, cfg = ArtifactStore(tmp_path / "Artifacts"), _nad_cfg()
        one, two = (_prior_artifact(store, cfg, name=name).id for name in ("a_prior", "b_prior"))
        page.show_view("saved")
        page.prior_picker.refresh()
        page.bounds_picker.restore_key("master.txt")
        page.prior_picker.restore_key(one)
        page.read_facts()
        said = page.prior_line.plain_text()
        assert said.startswith("Made ") and said.endswith(", for Nadrowski: 13 parameters, built on master.txt."), said
        assert [row.sentence.plain_text() for row in page.foot.facts.rows()] == [
            "Fine. Built on master.txt, the bounds file selected.",
            "Information. No saved simulation set was made with this prior."]
        page.bounds_picker.restore_key("master_tier1.txt")
        page.read_facts()
        assert page.foot.facts.rows()[0].sentence.plain_text() == (
            "Attention. Built on master.txt; the bounds file selected is master_tier1.txt. PRISM's own check compares "
            "only the model's parameters and their limits, which these two files share, so the load goes through.")
        page.bounds_picker.restore_key("master.txt")
        app, button, follower = QApplication.instance(), page.foot.main_button, style.current_follower()

        def loaded_by(key, prior, how, ends=lambda: stand_in.release(SimpleNamespace(kind="prior", id="", name="",
                                                                                      manifest=None))):
            page.prior_picker.restore_key(prior)
            page.read_facts()
            _settle()
            assert button.isEnabled(), page.foot.reason.text()
            if how == "keyboard":
                button.setFocus(Qt.TabFocusReason)
                _arrive(button, Qt.TabFocusReason)
                assert follower.target() is button
                QTest.keyClick(button, Qt.Key_Return)
            else:
                page.bounds_picker.combo.setFocus(Qt.TabFocusReason)
                _arrive(page.bounds_picker.combo, Qt.TabFocusReason)
                QTest.mouseClick(button, Qt.LeftButton)
            assert stand_in.started.wait(5)
            stand_in.started.clear()
            ends()
            turn_until(app, lambda: not controller().is_live(), 10.0)
            _settle()

        loaded_by("keyboard", one, "keyboard", lambda: stand_in.release(SimpleNamespace(kind="prior", id=one,
                                                                                        name="a_prior", manifest=None)))
        assert QApplication.focusWidget() is page.next_button and follower.target() is page.next_button
        loaded_by("keyboard", two, "keyboard", lambda: stand_in.fail(OSError("prior.pt cannot be read")))
        assert asked().title == "Error" and screen.session.inf_prior.id == one
        assert QApplication.focusWidget() is button, QApplication.focusWidget()
        loaded_by("mouse", two, "mouse", lambda: stand_in.release(SimpleNamespace(kind="prior", id=two,
                                                                                  name="b_prior", manifest=None)))
        assert QApplication.focusWidget() is window.shell.rail.row("prior") and follower.target() is None


def test_a_press_that_starts_nothing_hands_no_keyboard_to_a_later_load(tmp_path, monkeypatch):
    """A press by the keyboard is remembered while the run it started lives, and no longer. Answered "Keep the
    posterior", it started none; a load the chain starts later, with no key pressed, then hands the keyboard to no
    step."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core import orchestrator
    from core.artifacts import ArtifactStore
    from core.gui import style
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._look_helpers import _arrive, _settle
    from tests._runs import StandIn, turn_until
    stand_in = StandIn(word="Prior load")
    monkeypatch.setattr(orchestrator, "build_prior", stand_in)
    with _first_step_with_the_ring(tmp_path, monkeypatch) as (window, screen, first):
        _apply(first)
        window.go_to("inference", "prior")
        _settle()
        page = screen.prior_panel
        store, cfg = ArtifactStore(tmp_path / "Artifacts"), _nad_cfg()
        one, two = (_prior_artifact(store, cfg, name=name).id for name in ("a_prior", "b_prior"))
        held = SimpleNamespace(kind="posterior", id="post_old", name="retrain")
        screen.session.posterior = held
        page.show_view("saved")
        page.prior_picker.refresh()
        page.bounds_picker.restore_key("master.txt")
        page.prior_picker.restore_key(one)
        page.read_facts()
        _settle()
        app, button, follower = QApplication.instance(), page.foot.main_button, style.current_follower()
        button.setFocus(Qt.TabFocusReason)
        _arrive(button, Qt.TabFocusReason)
        assert follower.target() is button
        choose(monkeypatch, words="Keep the posterior")
        forget_asked()
        QTest.keyClick(button, Qt.Key_Return)
        _settle()
        assert asked().title == "Load this prior?" and not controller().is_live() and stand_in.calls == []
        assert page._by_keyboard is False, "a press that started no run is still taken for a press by the keyboard"
        assert page.load_record(two) is True
        assert stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="prior", id=two, name="b_prior", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        assert screen.session.inf_prior.id == two
        assert QApplication.focusWidget() is not page.next_button and follower.target() is not page.next_button


@pytest.mark.parametrize("load", ["returns its record", "fails"])
def test_after_a_load_how_the_run_went_opens_no_run_of_another_record(load, tmp_path, monkeypatch):
    """A build that finished leaves "How the run went" over its figure. A load of another prior that returns it closes
    the build's figure and the link with it: the page shows the record loaded alone, and no run's page holds its figure.
    A load that fails leaves the build's figure, the link, and the build's page holding that figure."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        stack, link = page.figure_stack, page.run_view.how_link
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.figure("Prior")
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        assert [stack.tabText(i) for i in range(stack.count())] == ["Prior"] and link.isVisibleTo(page.figures)
        two = _prior_artifact(default_store(), _nad_cfg(), name="other_prior").id
        page.show_view("saved")
        page.prior_picker.refresh()
        page.prior_picker.restore_key(two)
        page.read_facts()
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.figure("Prior (loaded)")
        turn_until(app, lambda: stack.count() == 2, 10.0)
        if load == "returns its record":
            stand_in.release(SimpleNamespace(kind="prior", id=two, name="other_prior", manifest=None))
        else:
            stand_in.fail(OSError("prior.pt cannot be read"))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        shown = [stack.tabText(i) for i in range(stack.count())]
        if load == "returns its record":
            assert shown == ["Prior (loaded)"] and not link.isVisibleTo(page.figures), shown
            return
        assert shown == ["Prior"] and link.isVisibleTo(page.figures), shown
        link.click()
        assert page.results.currentWidget() is page.run_view
        assert [stack.tabText(i) for i in range(stack.count()) if stack.isTabVisible(i)] == ["Prior"]


def test_a_store_change_during_a_build_reads_nothing_until_it_ends(tmp_path, monkeypatch):
    """Records may change the store while a build is live: the saved list is read again, which a run does not forbid,
    and nothing else is, neither the record's facts nor the sets nor the history, until the build has ended; it then
    reads once. A store change with no run live reads the page in sight at once."""
    from core.gui.run.controller import controller
    from tests._fixtures import readings_of_a_run
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        readings = []
        monkeypatch.setattr(page, "read_disk", lambda kept=page.read_disk: (readings.append(1), kept())[1])
        page.main_pressed()
        assert stand_in.started.wait(5)
        listed = []
        monkeypatch.setattr(page.prior_picker, "refresh",
                            lambda kept=page.prior_picker.refresh: (listed.append(1), kept())[1])
        calls = readings_of_a_run(monkeypatch)
        window._store_changed()
        assert listed == [1] and readings == [] and calls == [], (listed, readings, calls)
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert readings == [1]
        window._store_changed()
        assert readings == [1, 1], "a store change with no run live did not read the page in sight"


def test_a_load_by_code_reads_first_and_asks_nothing(tmp_path, monkeypatch):
    """A caller that asked its own question loads a saved prior through the page as its button does, the facts read
    first and no question asked: True when the load started, False when the facts block it, with nothing started. A
    load leaves the name and the note as typed for a build, and says nothing of a name."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        other_model = _prior_artifact(default_store(), _nad_cfg(), name="of_hopf", model="HOPF").id
        two = _prior_artifact(default_store(), _nad_cfg(), name="other").id
        assert page.load_record(other_model) is False and stand_in.calls == [], "a prior of another model was loaded"
        forget_asked()
        assert page.load_record(two) is True and all_asked() == []
        assert stand_in.started.wait(5) and stand_in.calls[-1][0][1:] == (two, False)
        stand_in.release(SimpleNamespace(kind="prior", id=two, name="other", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert screen.session.inf_prior.id == two and page.view == "saved"
        assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()
        page.show_view("build")
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try")


def test_the_two_lists_are_read_again_at_check_again_and_at_the_show_and_not_as_one_types(tmp_path, monkeypatch):
    """A bounds file put into the model's folder, and a prior written into the store by another program, which tells the
    window nothing, come into the two lists at "Check again", and as the page comes into sight again; the reading after
    the typing stops reads neither list. Each list keeps its selection."""
    import shutil
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._look_helpers import _settle
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        folder = tmp_path / "bounds"
        folder.mkdir()
        shutil.copyfile(page.bounds_picker.selected_path(), folder / "master.txt")
        page.bounds_picker.repoint(folder, "master.txt")
        page.read_facts()
        files, priors = page.bounds_picker.combo, page.prior_picker.combo

        def listed(combo):
            return [combo.itemData(at) for at in range(combo.count())]

        shutil.copyfile(folder / "master.txt", folder / "added.txt")
        made = _prior_artifact(default_store(), _nad_cfg(), name="from_the_command_line").id
        page.names.name.setText("retrain_3")
        page.read_timer.timeout.emit()                   # the reading after the typing stops
        assert "added.txt" not in listed(files) and made not in listed(priors), "the typing's reading read a list"
        page.foot.check_again.click()
        assert "added.txt" in listed(files) and made in listed(priors), (listed(files), listed(priors))
        assert page.bounds_picker.key() == "master.txt" and page.prior_picker.key() == ""
        shutil.copyfile(folder / "master.txt", folder / "another.txt")
        later = _prior_artifact(default_store(), _nad_cfg(), name="written_later").id
        window.go_to("home")
        _settle()
        window.go_to("inference", "prior")
        _settle()
        assert "another.txt" in listed(files) and later in listed(priors), (listed(files), listed(priors))
        assert page.bounds_picker.key() == "master.txt"


@pytest.mark.parametrize("then", ["another model applied", "a prior built for it on limits typed by hand"])
def test_the_bounds_file_remembered_comes_back_with_the_model_it_came_into_use_with(then, tmp_path, monkeypatch):
    """The bounds file is remembered with the model it came into use with. Another model applied after it, or a prior
    built for that model on limits typed by hand, which leave the file remembered as it was, change nothing of it: that
    model says nothing of the file, at the next launch either, and the model the file came into use with gives it back,
    in this session and kept from last time at the next."""
    from core.gui import settings as st
    from core.gui.remembered import BOUNDS_KEY, BOUNDS_MODEL_KEY, MODEL_KEY
    from core.gui.run.controller import controller
    from tests._look_helpers import _settle
    from tests._runs import turn_until
    app = qt_app()
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        first = screen.config_panel
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        window.go_to("inference", "model")
        _settle()
        first.model_combo.setCurrentText("HOPF")
        choose(monkeypatch, words="Apply and start a new session")
        _apply(first)
        assert page.bounds_picker.key() == "" and page.gone_line.isHidden()
        if then == "another model applied":
            first.model_combo.setCurrentText("NADROWSKI")
            _apply(first)
            assert page.bounds_picker.key() == "master.txt", "the file a build came into use on was not given back"
            first.model_combo.setCurrentText("HOPF")
            _apply(first)
        else:
            window.go_to("inference", "prior")
            _settle()
            page.show_view("build")
            page.bounds_picker.restore_key("cell.txt")
            page.typed_tick.setChecked(True)
            page.names.name.setText("hopf_typed")
            page.read_facts()
            assert page.foot.main_button.isEnabled(), page.foot.reason.text()
            page.main_pressed()
            assert stand_in.started.wait(5)
            stand_in.release(SimpleNamespace(kind="prior", id="p_hopf", name="hopf_typed", manifest=None))
            turn_until(app, lambda: not controller().is_live(), 10.0)
        qs = st.settings()
        qs.sync()
        assert (qs.value(MODEL_KEY), qs.value(BOUNDS_KEY), qs.value(BOUNDS_MODEL_KEY)) == (
            "HOPF", "master.txt", "NADROWSKI")
    again = build_inference()
    later = again.prior_panel
    again.config_panel.model_combo.setCurrentText("HOPF")
    again.config_panel.foot.main_button.click()
    assert later.bounds_picker.key() == "" and later.gone_line.isHidden()
    assert later.form.rows["bounds"].origin.text() == ""
    again.config_panel.model_combo.setCurrentText("NADROWSKI")
    again.config_panel.foot.main_button.click()
    assert later.bounds_picker.key() == "master.txt" and later.gone_line.isHidden()
    assert later.form.rows["bounds"].origin.text() == "kept from last time"


def test_the_configuration_in_use_says_its_units_and_its_force_scale(tmp_path, monkeypatch):
    """While a configuration is in use, what its units declare against the pipeline's own, and a scale of the force no
    drive can move, are said in "Before you start" as "Attention.", blocking nothing."""
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        cfg = screen.session.cfg
        cfg.check_unit_consistency = lambda: ["Units file declares frequency in 'Hz' but time in 'ms'."]
        cfg.rescale_params = {"f_scale": (None, (0.1, 10.0))}
        page.redraw()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows() if row.kind == "warn"]
        assert said[:2] == [
            "Attention. Units file declares frequency in 'Hz' but time in 'ms'.",
            "Attention. This bounds file lists f_scale, but a posterior trained on it reads a spontaneous recording only, "
            "where no drive acts: f_scale cannot change what is observed, so its estimate will be the prior's. Choose a "
            "bounds file with neither a forcing section nor f_scale for a spontaneous recording."]
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
