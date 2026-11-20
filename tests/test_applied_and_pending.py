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
from tests._window_helpers import _apply, _line, _pick, _units_file

#: The pages of Parameter inference built on the step page, by their files beside it.
REBUILT = ("config_tab.py", "prior_tab.py", "posterior_tab.py", "validate_tab.py", "infer_tab.py", "tsnpe_tab.py")
#: The files of a check's page, its form and its result, beside the Checks list's.
CHECK_FILES = ("check_page.py", "check_results.py", "check_forms.py")
#: What every step's "Before you start" says last while no plan is in use and every setting it reads is at its usual
#: value.
NO_PLAN = "Information. No plan. Every setting is at its usual value."


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
    ("chi_max_cycles", lambda page: page.form.rows["chi_max_cycles"].widget.setText("12.5"), "64 drive cycles"),
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
        assert foot.reason.text() == "Cannot start: this model cannot be used."
        first = foot.facts.rows()[0]
        assert (first.kind, first.sentence.plain_text()) == ("err", f"Error. {words}")
        page._build_config()
        assert screen.session.draft is None and all_asked() == []
        assert screen.step_states()[0].rail_words == "Now: choose a model"
        _pick(page.model_combo, "NADROWSKI")
        assert foot.main_button.isEnabled() and [row.sentence.plain_text() for row in foot.facts.rows()] == [NO_PLAN]


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
            session, observation_on_offer=screen.tsnpe_panel.observation_on_offer(), label=record_words.label_of)
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
    facts = [(fact.kind, fact.sentence.plain_text()) for fact in foot.facts.rows()]
    assert facts[0] == ("err", f"Error. {row.line.words}") and [kind for kind, _said in facts[1:]] == ["info"], facts
    assert facts[1][1].startswith("Information. No plan.") and row.label_text() not in facts[1][1], (
        "a setting in error was said as a departure from the usual")
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


# ── a units file of any folder, chosen ──
def _items(choice) -> list:
    """The words of every item of a list, in its order."""
    return [choice.itemText(at) for at in range(choice.count())]


def _units_row(page) -> tuple:
    """What the units row shows and what follows from it: its items, its choice, the line of the file under it, its own
    line, whether the step has nothing to apply and whether the main button can be pressed."""
    return (_items(page.units_choice), page.units_choice.currentData(), page.units_line.plain_text(),
            _line(page, "units"), page.nothing_to_apply(), page.foot.main_button.isEnabled())


def test_choose_a_units_file_stands_under_the_units_row_and_opens_at_the_models_units_folder(tmp_path, monkeypatch):
    """"Choose a units file…" stands under the units row, last, as wide as its words, and the Tab key reaches it after
    the list and the box of typed units. Pressed, it opens the system's chooser of a file to open, titled "Choose a units
    file", units files first and every file offered, at the units folder of the model shown on the form, read from the
    resources' folder of units as it is pressed, else at that folder itself. Cancelled, the row stands as it stood, and
    nothing is written to the settings. While a run is live, none opens."""
    from PySide6.QtWidgets import QSizePolicy
    from core import config
    from core.gui import settings
    from tests._dialogs import CHOOSER_CALLS, ChooserCall, answer_chooser
    from tests._runs import running
    units = tmp_path / "Units"
    (units / "hopf").mkdir(parents=True)
    _screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    row, button = page.form.rows["units"], page.choose_units_button
    assert button.text() == "Choose a units file…" and row.under[-1] is button and button.isVisibleTo(page)
    assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Maximum
    assert row.stops() == [page.units_choice, page.units_text, button]
    remembered = sorted(settings.settings().allKeys())
    monkeypatch.setattr(config, "UNITS_PATH", units)
    before = _units_row(page)
    answer_chooser(monkeypatch, "")
    _pick(page.model_combo, "HOPF")
    button.click()
    _pick(page.model_combo, "NADROWSKI")
    button.click()
    chooser = ("open", "Choose a units file")
    filter_ = "Units file (*.txt);;All files (*)"
    assert CHOOSER_CALLS == [ChooserCall(*chooser, str(units / "hopf"), filter_, ""),
                             ChooserCall(*chooser, str(units), filter_, "")], CHOOSER_CALLS
    assert _units_row(page) == before
    assert sorted(settings.settings().allKeys()) == remembered
    with running():
        page._choose_units_file()
    assert len(CHOOSER_CALLS) == 2, "a chooser opened while a run was live"


@pytest.mark.parametrize("model, bounds", [("NADROWSKI", "master.txt"), ("NADROWSKI", "master_tier1.txt"),
                                           ("HOPF", None)], ids=["Nadrowski", "the tier-1 box", "Hopf"])
def test_a_units_file_chosen_is_the_third_choice_and_what_is_applied_reads_it(model, bounds, tmp_path, monkeypatch):
    """A units file of a folder of its own, with units other than the model's, chosen: the units row's third choice,
    chosen, named by its file, its whole path, written as Windows writes it, its item's tip and what the draft will
    hold, so that a file a record names is the same file chosen again, its units under the row in the order of the
    alphabet, the
    fold's summary naming it, the row changed against the model's file applied, and nothing applied. Applied, the draft
    holds the file, "In use" names it, a plan would hold its units as words, the run's history its path, the settings
    file nothing, and the configuration made from it on the model's bounds file reads the file, its units and its bytes.
    A second file chosen takes the first one's place, one third choice in the list, changed against the first."""
    import hashlib
    from PySide6.QtCore import Qt
    from core import cli, config
    from core.artifacts.identity import _units_sha256
    from core.gui import settings
    from tests._dialogs import answer_chooser
    rig = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\nHz nm pN s\n")
    bench = _units_file(tmp_path / "bench", "bench units.txt", b"# Units\r\nnm ms pN kHz\r\n# the bench's own\r\n")
    screen, page = _first_step()
    _pick(page.model_combo, model)
    _apply(page)
    page.form.fold.set_open(True)
    model_words = page.units_line.plain_text().split(", from ")[0]
    remembered = sorted(settings.settings().allKeys())
    answer_chooser(monkeypatch, rig.as_posix())                  # as the system's chooser writes a path
    page.choose_units_button.click()
    assert _items(page.units_choice) == ["From the model's units file", "Typed here", "From the file rig units.txt"]
    assert page.units_choice.currentData() == "own" and page.units_choice.itemData(2, Qt.ToolTipRole) == str(rig)
    assert page.units_line.plain_text() == "Hz nm pN s, from the file rig units.txt"
    assert "Units From the file rig units.txt" in page.form.fold.summary.text(), page.form.fold.summary.text()
    assert _line(page, "units") == ("warn", f"Changed, not applied. In use: {model_words}, from the model's file.")
    assert screen.session.draft.units_override is None
    _apply(page)
    draft = screen.session.draft
    assert (draft.model, draft.units_override) == (model, str(rig)) and _line(page, "units") == ("", "")
    assert page.in_use.sentence.plain_text().endswith("Units Hz nm pN s, from the file rig units.txt.")
    assert sorted(page.plan_read(True)["units"].split()) == ["Hz", "nm", "pN", "s"]
    assert screen.posterior_panel.settings_entry()["options"]["units_override"] == str(rig)
    assert sorted(settings.settings().allKeys()) == remembered, "a file chosen, or its folder, was remembered"
    folder = config.BOUNDS_PATH / model.lower()
    bounds_file = folder / (bounds or sorted(path.name for path in folder.glob("*.txt"))[0])
    cfg = cli.make_sim_config(draft.model, draft.labels, draft.state_dep_drift, str(bounds_file),
                              units_override=draft.units_override, chi_mode=draft.chi_mode, hw=config.cpu_device())
    assert (cfg.sources["units"], sorted(cfg.units_dict)) == (str(rig), ["Hz", "nm", "pN", "s"])
    assert _units_sha256(cfg) == hashlib.sha256(rig.read_bytes()).hexdigest()
    answer_chooser(monkeypatch, str(bench))
    page.choose_units_button.click()
    assert _items(page.units_choice) == ["From the model's units file", "Typed here", "From the file bench units.txt"]
    assert page.units_line.plain_text() == "kHz ms nm pN, from the file bench units.txt"
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: Hz nm pN s, from the file rig units.txt.")


#: Units files the units row cannot take, each by how it is made in a folder of the test's own: its name, and what it
#: holds (None for a folder of that name, "gone" for no file at all), and the opening words of what the row says of it.
_REFUSED = {
    "a folder": ("units folder.txt", None, "The units file units folder.txt cannot be read: "),
    "bytes that are no text": ("units.bin", b"\xff\xfe\x00\x01", "The units file units.bin cannot be read: "),
    "gone since it was chosen": ("gone.txt", "gone", "The units file gone.txt cannot be read: File not found: "),
    "no units in it": ("master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n",
                       "The file master copy.txt holds no units: a units file lists them under \"# Units\"."),
    "a unit that does not resolve": ("odd units.txt", b"# Units\nnm notaunit ms pN\n",
                                     "Units from the file odd units.txt cannot be used: "),
}


@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_units_file_the_row_cannot_take_changes_nothing_of_the_row_and_says_why_under_it(case, tmp_path,
                                                                                         monkeypatch):
    """A file chosen that cannot be read, or whose units cannot be used, is read at once and taken nowhere: the units
    row stands as it stood, its third choice still the file applied before, and the reason stands under it and among
    the facts as a refusal of the row does; beside the greyed main button, that the units file chosen cannot be used,
    since no start was made. An edit of another row leaves it, and once the page reads again it stops the start no
    longer, its sentence standing as a warning until the row is edited. A start refused on the same row still says it
    was."""
    from core.refusals import Refusal
    from tests._dialogs import answer_chooser
    name, data, said = _REFUSED[case]
    rig = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\nHz nm pN s\n")
    chosen = tmp_path / "chosen" / name
    chosen.parent.mkdir()
    if data is None:
        chosen.mkdir()
    elif data != "gone":
        chosen.write_bytes(data)
    screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    answer_chooser(monkeypatch, str(rig))
    page.choose_units_button.click()
    _apply(page)
    _pick(page.rot_choice, False)
    items, choice, line, _own, _nothing, _enabled = before = _units_row(page)
    assert before[3:] == (("", ""), False, True), before
    answer_chooser(monkeypatch, str(chosen))
    page.choose_units_button.click()
    kind, words = _line(page, "units")
    assert (kind, words.startswith(said)) == ("err", True), (kind, words)
    assert _units_row(page)[:3] == (items, choice, line) and screen.session.draft.units_override == str(rig)
    assert not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: the units file chosen cannot be used."
    facts = [(fact.kind, fact.sentence.plain_text()) for fact in page.foot.facts.rows()]
    assert facts[0] == ("err", f"Error. {words}"), facts
    _pick(page.kind_choice, True)                      # what cures it may stand in another row
    assert _line(page, "units") == ("err", words) and not page.foot.main_button.isEnabled()
    page.read_facts()
    got = (_line(page, "units"), page.foot.main_button.isEnabled(), page.foot.reason.text())
    assert got == (("warn", words), True, ""), got
    _pick(page.units_choice, "file")
    assert _line(page, "units") == ("warn", "Changed, not applied. In use: Hz nm pN s, from the file rig units.txt.")
    assert page.foot.main_button.isEnabled()
    page._refusal(Refusal("The units were refused.", field="units"))
    assert page.foot.reason.text() == "Cannot start yet: the last start was refused."


def _refused_units_row(state, tmp_path, monkeypatch):
    """The first step, in a window in sight, applied, its units row put in ``state`` against what the step holds ("typed units", "the model's
    file with typed units held", "another units file"), and a file that holds no units chosen and refused: (screen,
    page, the sentence under the row, the file refused, the units a press of the main button applies)."""
    from tests._dialogs import answer_chooser
    rig = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\nHz nm pN s\n")
    bad = _units_file(tmp_path / "bad", "master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n")
    _window, screen, page = _shown_first_step(tmp_path)
    _apply(page)
    page.form.fold.set_open(True)
    if state == "typed units":
        _pick(page.units_choice, "typed")
        page.units_text.setText("kHz ms nm pN")
        applied = ("kHz", "ms", "nm", "pN")
    elif state == "the model's file with typed units held":
        _pick(page.units_choice, "typed")
        page.units_text.setText("nm ms pN kHz")
        _apply(page)
        _pick(page.units_choice, "file")
        applied = None
    else:
        answer_chooser(monkeypatch, str(rig))
        page.choose_units_button.click()
        applied = str(rig)
    assert page.nothing_to_apply() is False
    answer_chooser(monkeypatch, str(bad))
    page.choose_units_button.click()
    kind, words = _line(page, "units")
    assert (kind, words) == ("err", "The file master copy.txt holds no units: a units file lists them under "
                                    "\"# Units\"."), (kind, words)
    return screen, page, words, bad, applied


@pytest.mark.parametrize("reading", ["the page's timer", "a reading asked for"])
@pytest.mark.parametrize("state", ["typed units", "the model's file with typed units held", "another units file"])
def test_a_refused_units_files_sentence_stands_over_a_change_not_applied_until_the_row_is_edited(state, reading,
                                                                                               tmp_path, monkeypatch):
    """A units file chosen and refused keeps its sentence under the units row while the row holds a change not applied,
    as it does while the row holds what the step holds: the page reading again (its timer after an edit of another
    row, or a reading asked for) turns it from an error into a warning and says no "Changed, not applied" in its
    place, for it is the one place the sentence is said. The change not applied is still counted, in "In use", on the
    fold and on the rail. The main button applies what the row showed, never the refused file, and the sentence still
    stands; at the next edit of the units row it goes, and "Changed, not applied" takes its place, or nothing when the
    row is back at what the step holds."""
    screen, page, words, bad, applied = _refused_units_row(state, tmp_path, monkeypatch)
    facts = [(fact.kind, fact.sentence.plain_text()) for fact in page.foot.facts.rows()]
    assert facts[0] == ("err", f"Error. {words}") and not page.foot.main_button.isEnabled(), facts
    assert page.foot.reason.text() == "Cannot start yet: the units file chosen cannot be used."
    _pick(page.rot_choice, not page.rot_choice.currentData())
    if reading == "the page's timer":
        page.read_timer.timeout.emit()
    else:
        page.read_facts()
    assert _line(page, "units") == ("warn", words), _line(page, "units")
    assert page.foot.main_button.isEnabled() and page.foot.reason.text() == ""
    assert not any(words in fact.sentence.plain_text() for fact in page.foot.facts.rows())
    assert page.in_use.pending.sentence.plain_text() == "Attention. 2 choices below are changed and not applied."
    assert "2 changed" in page.form.fold.button.text(), page.form.fold.button.text()
    assert screen.pending() == {"model"}
    page.read_facts()
    assert _line(page, "units") == ("warn", words), "a second reading took the sentence away"
    _apply(page)
    assert str(bad) not in str(screen.session.draft.units_override)
    assert screen.session.draft.units_override == applied
    assert _line(page, "units") == ("warn", words), _line(page, "units")
    held = page.units_choice.currentData()
    _pick(page.units_choice, "file" if held != "file" else "typed")
    page.read_facts()
    said = _line(page, "units")
    assert said[0] == "warn" and said[1].startswith("Changed, not applied. In use: "), said
    _pick(page.units_choice, held)
    page.read_facts()
    assert _line(page, "units") == ("", "")


def test_a_refused_start_gives_way_to_a_change_not_applied_where_a_refused_units_file_does_not(tmp_path, monkeypatch):
    """A start refused for the units gives way when the page reads again: the units row holds a change not applied, so
    the sentence the refusal marked it with is replaced by "Changed, not applied. In use: …", for that sentence was
    never a file's. A units file chosen and refused, with the row in that same state, keeps its own sentence at the
    reading, as a warning, and says no "Changed, not applied" in its place."""
    from core.refusals import Refusal
    from tests._dialogs import answer_chooser
    _screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    _pick(page.units_choice, "typed")
    page.units_text.setText("kHz ms nm pN")
    page._refusal(Refusal("The units were refused.", field="units"))
    assert _line(page, "units") == ("err", "The units were refused.")
    page.read_facts()
    said = _line(page, "units")
    assert said[0] == "warn" and said[1].startswith("Changed, not applied. In use: "), said
    bad = _units_file(tmp_path / "bad", "master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n")
    _pick(page.units_choice, "file")
    _pick(page.units_choice, "typed")
    page.units_text.setText("kHz ms nm pN")
    answer_chooser(monkeypatch, str(bad))
    page.choose_units_button.click()
    kind, words = _line(page, "units")
    assert (kind, words.startswith("The file master copy.txt holds no units")) == ("err", True), (kind, words)
    page.read_facts()
    assert _line(page, "units") == ("warn", words), _line(page, "units")


def test_a_change_of_model_leaves_a_refused_units_files_sentence_until_the_row_is_edited(tmp_path, monkeypatch):
    """A model chosen fills the box of typed units from its units file; while the units row is not on "Typed here" that
    box is out of sight and what fills it is no edit of the row, so a refused units file's sentence stays under the row,
    an error until the page reads again and a warning after. While units are typed and still as a file filled them, the
    filling does change the units the row holds: the box holds the new model's units and the sentence goes."""
    from tests._dialogs import answer_chooser
    bad = _units_file(tmp_path / "bad", "master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n")
    for typed in (False, True):
        _screen, page = _first_step()
        _apply(page)
        page.form.fold.set_open(True)
        if typed:
            _pick(page.units_choice, "typed")
        answer_chooser(monkeypatch, str(bad))
        page.choose_units_button.click()
        words = _line(page, "units")[1]
        assert _line(page, "units")[0] == "err" and words.startswith("The file master copy.txt holds no units")
        before = page.units_text.text()
        _pick(page.model_combo, "BP")
        filled = page._units_of("BP")[0]
        assert filled != before, "the two models' units files say the same"
        if typed:
            assert page.units_text.text() == filled
            assert _line(page, "units")[1] != words, "the sentence stood over a row whose units had changed"
            continue
        assert _line(page, "units") == ("err", words), _line(page, "units")
        page.read_facts()
        assert _line(page, "units") == ("warn", words), _line(page, "units")
        assert page.units_text.text() == filled and page.units_choice.currentData() == "file"


def test_choose_a_units_file_pressed_again_takes_away_what_it_said_of_the_file_it_refused(tmp_path, monkeypatch):
    """A later press of "Choose a units file…" takes away the sentence its own earlier press left under the units row,
    whether it still stops the start or the page has read since: cancelled, the row stands as it stood before the
    refused file, the main button back; a usable file is taken; a file refused again is said again. A refused start's
    sentence is no press's to take away: cancelled, it stays, and so does the greyed button."""
    from core.refusals import Refusal
    from tests._dialogs import answer_chooser
    rig = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\nHz nm pN s\n")
    bad = _units_file(tmp_path / "bad", "master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n")
    _screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    _pick(page.rot_choice, False)
    before = _units_row(page)
    refused = ("err", "The file master copy.txt holds no units: a units file lists them under \"# Units\".")
    for read_since in (False, True):
        answer_chooser(monkeypatch, str(bad))
        page.choose_units_button.click()
        assert (_line(page, "units"), page.foot.main_button.isEnabled()) == (refused, False), read_since
        if read_since:
            page.read_facts()
            assert _line(page, "units") == ("warn", refused[1]) and page.foot.main_button.isEnabled()
        answer_chooser(monkeypatch, "")
        page.choose_units_button.click()
        assert (_units_row(page), page.foot.reason.text()) == (before, ""), (read_since, _units_row(page))
    answer_chooser(monkeypatch, str(bad))
    page.choose_units_button.click()
    answer_chooser(monkeypatch, str(rig))
    page.choose_units_button.click()
    assert page.units_choice.currentData() == "own" and _line(page, "units")[0] == "warn", _line(page, "units")
    page._refusal(Refusal("The units were refused.", field="units"))
    answer_chooser(monkeypatch, "")
    page.choose_units_button.click()
    assert _line(page, "units") == ("err", "The units were refused.") and not page.foot.main_button.isEnabled()
    assert page.foot.reason.text() == "Cannot start yet: the last start was refused."


def test_a_units_file_refused_while_units_are_typed_outlines_no_box(tmp_path, monkeypatch):
    """While units are typed, a file chosen and refused marks the units row and leaves the box of typed units without
    the outline of an error, for what it holds is not what is refused; a refused start of the row outlines it."""
    from core.refusals import Refusal
    from tests._dialogs import answer_chooser
    bad = _units_file(tmp_path / "bad", "master copy.txt", b"# Parameters\nk_gs = 1.0 in (0.5, 2.0)\n")
    _screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    _pick(page.units_choice, "typed")
    answer_chooser(monkeypatch, str(bad))
    page.choose_units_button.click()
    assert _line(page, "units")[0] == "err" and page.units_text.property("state") == "", (
        _line(page, "units"), page.units_text.property("state"))
    page._refusal(Refusal("The units were refused.", field="units"))
    assert page.units_text.property("state") == "error", "a refused start of the units left their box unoutlined"


def test_a_units_file_holding_no_units_is_said_to_hold_none_when_a_chain_applies_it(tmp_path):
    """A units file the engine's reader finds no units in, applied as a chain applies a record's, is refused on the
    units row in words of its own, as it is when chosen, and nothing is applied."""
    screen, page = _first_step()
    _apply(page)
    draft = screen.session.draft
    empty = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\n")
    assert not page.apply_quietly("NADROWSKI", {"chi_mode": False, "units_override": str(empty)})
    assert screen.session.draft is draft, "a units file that holds no units was applied"
    said = ("err", "The file rig units.txt holds no units: a units file lists them under \"# Units\".")
    assert _line(page, "units") == said, _line(page, "units")


def test_a_units_library_error_with_no_words_still_refuses_a_units_file(tmp_path, monkeypatch):
    """An error of the units library that says nothing still refuses a units file, chosen or applied as a chain applies
    a record's, named by what it is."""
    from core import cli
    from tests._dialogs import answer_chooser
    rig = _units_file(tmp_path / "rig", "rig units.txt", b"# Units\nHz nm pN s\n")
    screen, page = _first_step()
    _apply(page)
    page.form.fold.set_open(True)
    draft = screen.session.draft

    def no_words(units):
        raise ValueError()

    monkeypatch.setattr(cli, "units_to_factors", no_words)
    said = ("err", "Units from the file rig units.txt cannot be used: ValueError")
    answer_chooser(monkeypatch, str(rig))
    page.choose_units_button.click()
    assert (page.units_choice.currentData(), _line(page, "units")) == ("file", said), _line(page, "units")
    assert not page.apply_quietly("NADROWSKI", {"chi_mode": False, "units_override": str(rig)})
    assert screen.session.draft is draft and _line(page, "units") == said, _line(page, "units")


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
    touched, and nothing is applied: the step holds nothing. Applying keeps the word on what was not touched. Every
    other row a plan fills says it holds its usual value."""
    _remember(**{"inference_config/model": "HOPF", "inference_config/chi_mode": "true"})
    screen, page = _first_step()
    rows = page.form.rows
    assert (page.model_combo.currentText(), page.kind_choice.currentData()) == ("HOPF", True)
    assert [key for key, row in rows.items() if row.origin.text()] == [
        "model", "chi_mode", "chi_n_freqs", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds",
        "reparam_rotate", "units"]
    assert rows["model"].origin.text() == rows["chi_mode"].origin.text() == "kept from last time"
    assert {rows[key].origin.text() for key in ("chi_n_freqs", "chi_k_pad", "chi_max_cycles", "reparam_rotate",
                                                 "units")} == {"usual"}
    assert screen.session.draft is None and page.in_use.line_text() == "In use: nothing yet."
    assert page.in_use.sentence.plain_text() == "Choose the model and the kind of observation, then apply them."
    _pick(page.model_combo, "NADROWSKI")
    assert rows["model"].origin.text() == "" and rows["chi_mode"].origin.text() == "kept from last time"
    _apply(page)
    assert rows["chi_mode"].origin.text() == "kept from last time" and rows["model"].origin.text() == ""
    _pick(page.kind_choice, False)
    _pick(page.kind_choice, True)
    assert rows["chi_mode"].origin.text() == "changed from the usual", "a kind touched and put back is still kept"


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
    assert (rows["model"].origin.text(), rows["chi_mode"].origin.text()) == ("", "usual")


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
    """The two keys are written when the step is applied, by the one writer, together, with the plan last used with the
    model, blank with no plan in use; a draft that is not the window's own writes nothing it could not read back; and
    the step writes nothing as the window closes."""
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
    assert written == [[("inference_config/model", "HOPF"), ("inference_config/chi_mode", True),
                        ("plans/last_used/hopf", ""), ("plans/last_used_at/hopf", "")]]
    qs = settings.settings()
    assert (qs.value("inference_config/model"), settings.get_bool(qs, "inference_config/chi_mode", False)) == (
        "HOPF", True)
    screen.new_draft(SimpleNamespace(model="BP", chi_mode="yes"))

    class NoName:
        """A model that is no string, as a draft that is not the window's own may hold: it says only its folder."""

        def lower(self):
            return "bp"

    screen.new_draft(SimpleNamespace(model=NoName(), chi_mode=True))
    assert written[1:] == [
        [("inference_config/model", "BP"), ("plans/last_used/bp", ""), ("plans/last_used_at/bp", "")],
        [("inference_config/chi_mode", True)]], written
    page.save_settings(qs)
    qs.sync()
    again = settings.settings()
    assert sorted(again.allKeys()) == ["inference_config/chi_mode", "inference_config/model", "plans/last_used/bp",
                                       "plans/last_used/hopf", "plans/last_used_at/bp", "plans/last_used_at/hopf"]
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
    """A rebuilt page asks and tells through the one shared question: neither the step page nor any page built on it,
    nor a check's page, form or result, names the toolkit's message box, or keeps a log pane of its own. The files read
    are every page's built on the step page and every file of a check's beside them."""
    folder = Path(__file__).resolve().parents[1] / "core" / "gui" / "panels" / "inference"
    from tests._fixtures import step_pages
    assert sorted({file for file, _step in step_pages().values()}) == sorted(REBUILT), step_pages()
    assert sorted(path.name for path in folder.glob("check_*.py")) == sorted(CHECK_FILES), "a check's file not read"
    for name in ("step_page.py", *REBUILT, *CHECK_FILES):
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
                 "before you rely on its posterior."), ("info", NO_PLAN)]
    assert page.foot.main_button.isEnabled() and page.foot.reason.text() == ""
    usable[0] = False
    page.redraw()
    assert [row.kind for row in page.foot.facts.rows()] == ["err", "info"]
    _pick(page.model_combo, "NADROWSKI")
    assert [row.sentence.plain_text() for row in page.foot.facts.rows()] == [NO_PLAN]


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
    note stand as typed, the rail and "In use" say what they said, and the figures of the prior in use stand as they
    stood, whatever the build drew before it ended. The checks are read first, as a launch that put these records in use
    by its runs has read them, since the build's end reads them."""
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        screen.read_checks()
        said = (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("prior").accessibleName())
        stack = page.figure_stack
        stack.add_figure("The prior in use", b"")
        if end != "withdrawn":
            stand_in.figure("The build's own")          # drawn as the build starts, before it ends
        _press_and_end(page, stand_in, end, monkeypatch)
        assert [stack.tabText(at) for at in range(stack.count())] == ["The prior in use"], (
            end, [stack.tabText(at) for at in range(stack.count())])
        assert screen.session is session
        assert all(now is then for now, then in zip((session.cfg, session.inf_prior, session.posterior,
                                                     session.observation), held)), end
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try"), end
        assert (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("prior").accessibleName()) == said, end


def test_a_build_that_returns_its_prior_shows_its_own_figures_alone(tmp_path, monkeypatch):
    """The figures of the prior in use stand out of sight on the build's own page while it runs, and go once the build
    has returned its prior: the page then shows the new prior's figures alone."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        stack = page.figure_stack
        stack.add_figure("The prior in use", b"")
        stand_in.figure("The build's own")
        page._main_clicked()
        assert stand_in.started.wait(5)
        turn_until(app, lambda: stack.count() == 2)
        during = [stack.tabText(at) for at in range(stack.count())]
        written = _prior_artifact(default_store(), _nad_cfg(), name="retrain_2").id
        stand_in.release(SimpleNamespace(kind="prior", id=written, name="retrain_2",
                                         manifest=default_store().get("prior", written)))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert during == ["The prior in use", "The build's own"], during
        assert [stack.tabText(at) for at in range(stack.count())] == ["The build's own"]


def test_a_prior_comes_into_use_only_when_its_run_returns(tmp_path, monkeypatch):
    """The press holds nothing new. The prior the run returns comes into use with the very configuration the press
    handed the run, while the run is still live: what depended on the prior held goes with it, the states are said once,
    on the window's thread, and a training continued with the session is one made with what it holds; step 5 is told of
    the configuration once, as it comes into use. At the run's end the saved list is read again and moved to the new
    prior, the name and the note are emptied, and the way to Training stands."""
    import threading
    from core.artifacts import default_store
    from core.gui.run import retry
    from tests._fixtures import _nad_cfg, _prior_artifact
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
        written = _prior_artifact(default_store(), _nad_cfg(), name="retrain_2").id     # what the stage writes
        later = screen.infer_panel
        assert not hasattr(later, "on_config_built"), "step 5 is still handed the configuration"
        later.read_facts()                                    # step 5 read once, before the prior comes back
        made = SimpleNamespace(kind="prior", id=written, name="retrain_2",
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
        assert later.stale, "step 5, not in sight, is not marked to read the configuration that came into use"
        later.read_facts()
        assert later._held()[0] is candidate and later._held_for[0] is candidate, (
            "step 5 does not read the configuration in use at its next reading")
        assert page.prior_picker.key() == written, "the saved list was not moved to the prior just made"
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


@contextlib.contextmanager
def _second_step_with_the_ring(tmp_path, monkeypatch):
    """The second step in the whole program, whose ring follows the keyboard as it does for a person, in sight off the
    screen and in front, on the test's own clock and its own store: the first step applied, a configuration, a prior, a
    posterior and an observation held, the build shown on the master bounds file, a name typed, and the stage's build a
    stand-in run: (window, screen, page, the stand-in run)."""
    from core import orchestrator
    from tests._fixtures import whole_program
    from tests._look_helpers import _settle
    from tests._runs import Clock, StandIn, at, controller_on
    from tests._window_helpers import in_front, shown_off_screen
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "model")
        _settle()
        screen = window.part("inference")
        _apply(screen.config_panel)
        cfg, prior, posterior, observation = _held_records()
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior, posterior=posterior,
                                    observation=observation)
        screen.refresh_gates()
        window.go_to("inference", "prior")
        _settle()
        page = screen.prior_panel
        page.show_view("build")
        page.bounds_picker.restore_key("master.txt")
        page.names.name.setText("retrain_2")
        page.read_facts()
        stand_in = StandIn(word="Prior build")
        monkeypatch.setattr(orchestrator, "build_prior", stand_in)
        with in_front(window):
            yield window, screen, page, stand_in
        window.hide()


def test_a_held_enter_on_build_a_new_prior_leaves_its_question_for_a_fresh_press(tmp_path, monkeypatch):
    """With a posterior held and the ring on "Build a new prior (…)", Enter presses it and its question opens where the
    key's repeats land: they answer nothing, and the question stands; Enter let go and pressed again gives the answer it
    preselects, "Keep the posterior", so nothing starts and the posterior stays held. A fresh press on the button asks
    again, so the window's own Enter still presses the button with the ring."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from core.gui.run.controller import controller
    from tests._dialogs import held_enter
    from tests._look_helpers import _arrive
    with _second_step_with_the_ring(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        button, held = page.foot.main_button, screen.session.posterior
        assert button.isEnabled() and button.text().startswith("&Build a new prior"), button.text()
        seen = []
        held_enter(monkeypatch, seen)
        for _ in range(2):
            button.setFocus(Qt.TabFocusReason)
            _arrive(button, Qt.TabFocusReason)
            QTest.keyClick(button, Qt.Key_Return)
        assert [(said.title, said.default) for said in all_asked()] == [
            ("Build a new prior?", "Keep the posterior")] * 2
        assert seen == [(True, False)] * 2, "a held Enter answered the question, or Enter afresh did not"
        assert not controller().is_live() and stand_in.calls == [] and screen.session.posterior is held


@contextlib.contextmanager
def _third_step_with_the_ring(tmp_path, monkeypatch):
    """The third step in the whole program, whose ring follows the keyboard as it does for a person, in sight off the
    screen and in front, on the test's own clock and its own store: the first step applied, a real prior of the master
    bounds file in use with its configuration, the batches and the simulations per batch typed, a name typed, and the
    stage's training a stand-in run: (window, screen, page, the stand-in run)."""
    from core import orchestrator
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _prior_artifact, whole_program
    from tests._look_helpers import _settle
    from tests._runs import Clock, StandIn, at, controller_on
    from tests._window_helpers import in_front, shown_off_screen
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "Artifacts"))
    with whole_program() as (_app, window):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "model")
        _settle()
        screen = window.part("inference")
        _apply(screen.config_panel)
        cfg = _nad_cfg()
        prior = default_store().load_prior(cfg, _prior_artifact(default_store(), cfg, name="retrain_prior").id)
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior)
        screen.prior_panel.bounds_picker.restore_key("master.txt")
        screen.refresh_gates()
        window.go_to("inference", "training")
        _settle()
        page = screen.posterior_panel
        _a_budget(page, runs="12", cap="8")
        page.names.name.setText("retrain_2")
        page.read_facts()
        stand_in = StandIn(word="Training")
        monkeypatch.setattr(orchestrator, "build_posterior", stand_in)
        with in_front(window):
            yield window, screen, page, stand_in
        window.hide()


@pytest.mark.parametrize("which", ["a kept training", "from zero", "a narrowed posterior", "the stage's refusal"])
def test_each_question_of_the_training_page_waits_for_enter_pressed_afresh(which, tmp_path, monkeypatch):
    """With the ring on the Training page's main button, Enter presses it, and the question it opens, or the refusal the
    stage gives as its run starts, stands where the key's repeats land: they answer nothing; Enter let go and pressed
    again gives the answer it preselects, "Cancel" or "OK", so nothing starts that was not started."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._dialogs import held_enter
    from tests._look_helpers import _arrive
    from tests._runs import turn_until
    with _third_step_with_the_ring(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        if which == "a kept training":
            _a_kept_training(page)
        elif which == "from zero":
            _a_set_beside(page, 40, n_runs=40)
        elif which == "a narrowed posterior":
            _trained, narrowed = _saved_posteriors(page)
            _to_the_load_view(page)
            page.post_picker.restore_key(narrowed)
        page.read_facts()
        button = page.foot.main_button
        assert button.isEnabled(), page.foot.reason.text()
        seen = []
        held_enter(monkeypatch, seen)
        button.setFocus(Qt.TabFocusReason)
        _arrive(button, Qt.TabFocusReason)
        QTest.keyClick(button, Qt.Key_Return)
        if which == "the stage's refusal":
            turn_until(qt_app(), lambda: len(stand_in.calls) == 1, 10.0)
            stand_in.fail(Refusal("A posterior named 'retrain_2' was made meanwhile.", field="name"))
            turn_until(qt_app(), lambda: not controller().is_live() and seen, 10.0)
        titles = {"a kept training": "Start a new training?", "from zero": "Start a new simulation set?",
                  "a narrowed posterior": "Load a posterior from a narrowing round?",
                  "the stage's refusal": "Check your inputs"}
        assert [said.title for said in all_asked()] == [titles[which]]
        assert seen == [(True, False)], "a held Enter answered the question, or Enter afresh did not"
        assert len(stand_in.calls) == (1 if which == "the stage's refusal" else 0)
        assert not controller().is_live()


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
            "Information. No saved simulation set was made with this prior.", NO_PLAN]
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


def test_before_you_start_reads_the_history_the_sets_and_the_settings_in_the_build_view(tmp_path, monkeypatch):
    """"Before you start" as the build view reads it, word for word: what the build costs, from the settings typed and
    the candidate's step; how long the last build on this computer took, from the run history, said on the button too;
    and the saved simulation set of the model, which a new prior cannot continue."""
    from datetime import datetime, timedelta, timezone
    from core.artifacts import default_store, write_simulation_manifest
    from core.gui.run.controller import controller
    from core.SBI.training_checkpoint import identity_digest
    from tests._fixtures import write_history
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        started = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        write_history(controller().folder(), id="20261008T120000-prior-build", word="Prior build", name="retrain_1",
                      outcome="finished", started=started, ended=started + timedelta(seconds=92))
        identity = {"format": "training-rows/2", "model": "NADROWSKI", "prior_fingerprint": "f" * 16, "n_runs": 4,
                    "truncation": None}
        write_simulation_manifest(default_store().kind_dir("simulation") / identity_digest(identity), identity,
                                  batches_done=4, complete=True)
        rows = page.form.rows
        rows["num_iterations"].widget.setText("3")
        rows["sweep_batch"].widget.ticked_box.setChecked(False)
        rows["sweep_batch"].widget.field.setText("64")
        rows["max_sets"].widget.setText("100")
        rows["stability_units"].widget.setText("2")
        page.read_facts()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert said == [
            "Information. Tests 192 candidates (3 rounds of 64), each integrated over 80 steps, then walks outward "
            "until 100 stable sets are collected.",
            "Information. About a minute and a half: the last build on this computer took 1 min 32 s.",
            "Information. A new prior cannot continue a saved simulation set: the saved set for this model was made "
            "with another prior.",
            "Information. No plan. 4 settings differ from the usual: Sweep rounds 3 (usual 50); Candidates per round "
            "64 (usual automatic); Stable sets to collect 100 (usual 175,000); Stability test length 2 (usual "
            "1000)."], said
        assert page.foot.main_button.text() == "&Build a new prior (about a minute and a half)"


def test_before_you_start_reads_the_record_and_its_sets_in_the_saved_view(tmp_path, monkeypatch):
    """"Before you start" as the saved view reads it, word for word: a saved prior built on the bounds file selected as
    it was before that file was edited, its box no longer the file's, said as a load PRISM will refuse; and the saved
    simulation set made with it, found by its fingerprint."""
    import json
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _prior_artifact, write_simulation_set
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        store, cfg = default_store(), _nad_cfg()
        lows = [bounds[0] for _value, bounds in cfg.params_dict.values()]
        lows[0] = lows[0] * 2
        made = _prior_artifact(store, cfg, name="older_box", lows=lows).id
        manifest = next(store.kind_dir("prior").glob(f"*__{made}")) / "manifest.json"
        written = json.loads(manifest.read_text(encoding="utf-8"))
        written["inputs"]["bounds"]["sha256"] = "0" * 64           # the file it was built on, edited since
        manifest.write_text(json.dumps(written, indent=1), encoding="utf-8")
        identity = {"format": "training-rows/2", "n_runs": 4, "truncation": None,
                    "prior_fingerprint": store.get("prior", made).fingerprints["gmm"]}
        write_simulation_set(store.kind_dir("simulation"), identity, 4, complete=True)
        page.show_view("saved")
        page.read_facts()
        page.prior_picker.restore_key(made)
        page.read_facts()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert said == [
            "Attention. Built on master.txt as it was then; the file has been edited since. Its parameters or their "
            "limits have changed with it, so PRISM will refuse the load.",
            "Information. 1 saved simulation set was made with this prior; it holds 4 batches.", NO_PLAN], said


@pytest.mark.parametrize("way", ["Back to the form", "the past view's way back"])
def test_after_a_load_how_the_run_went_stays_away_until_a_build_ends(way, tmp_path, monkeypatch):
    """A load that returns its record withdraws "How the run went" whatever stands in the column as it starts. A load by
    code while the build's page is open leaves that page up, without the build's figure; its way back, "Back to the
    form" or the past view's, gives the figures again, the loaded record's alone, with no link. The next build that ends
    brings the link back, and "Back to the form" after it keeps it."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        stack, link = page.figure_stack, page.run_view.how_link

        def ran(figure, made):
            assert stand_in.started.wait(5)
            stand_in.started.clear()
            stand_in.figure(figure)
            stand_in.release(made)
            turn_until(app, lambda: not controller().is_live(), 10.0)
            _settle_events()

        page.main_pressed()
        ran("Prior", SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        assert link.isVisibleTo(page.figures)
        link.click()
        assert page.results.currentWidget() is page.run_view
        two = _prior_artifact(default_store(), _nad_cfg(), name="other_prior").id
        assert page.load_record(two) is True
        ran("Prior (loaded)", SimpleNamespace(kind="prior", id=two, name="other_prior", manifest=None))
        assert page.results.currentWidget() is page.run_view, "the load took the column"
        if way == "Back to the form":
            page.run_view.back_button.click()
        else:
            page.show_past_view()
            page.past_view.back.emit()
        _settle_events()
        assert page.results.currentWidget() is page.figures
        assert [stack.tabText(i) for i in range(stack.count())] == ["Prior (loaded)"]
        assert not link.isVisibleTo(page.figures), f"the link came back by {way}"
        page.show_view("build")
        page.names.name.setText("retrain_3")
        page.read_facts()
        page.main_pressed()
        ran("Prior", SimpleNamespace(kind="prior", id="p_third", name="retrain_3", manifest=None))
        assert link.isVisibleTo(page.figures), "the next build's end did not bring the link back"
        link.click()
        page.run_view.back_button.click()
        _settle_events()
        assert link.isVisibleTo(page.figures), "the link went at the next build's own way back"


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
    """A caller that asked its own question loads a saved prior through the page as its button does, the lists and the
    facts read first and no question asked: True when the load started; False, with nothing started, when the facts
    block it, as for a prior made for another model and for one that cannot be read, each selected and said. A prior
    saved by another program since the page last read its lists is found. A load leaves the name and the note as typed
    for a build, and says nothing of a name."""
    from core.artifacts import ArtifactStore, default_store
    from core.artifacts.store import StoreError
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        store = default_store()
        other_model = _prior_artifact(store, _nad_cfg(), name="of_hopf", model="HOPF").id
        unreadable = _prior_artifact(store, _nad_cfg(), name="unreadable").id
        kept_get = ArtifactStore.get

        def get(held, kind, ref, *args, **kwargs):
            """What the store raises for a record gone or damaged since it was listed."""
            if ref == unreadable:
                raise StoreError(f"no complete prior artifact named or id'd {ref!r}")
            return kept_get(held, kind, ref, *args, **kwargs)

        monkeypatch.setattr(ArtifactStore, "get", get)
        blocks = ((other_model, "Error. This prior was made for the model Hopf."),
                  (unreadable, f"Error. The prior cannot be read: StoreError: no complete prior artifact named or id'd "
                               f"'{unreadable}'"))
        for ref, said in blocks:
            assert page.load_record(ref) is False and stand_in.calls == [], ref
            assert page.prior_picker.key() == ref, "the load was refused before the facts were read"
            facts = [row.sentence.plain_text() for row in page.foot.facts.rows()]
            assert facts[0] == said and not page.foot.main_button.isEnabled(), facts
        two = _prior_artifact(store, _nad_cfg(), name="other").id          # saved since the page last read its lists
        forget_asked()
        assert page.load_record(two) is True and all_asked() == []
        assert stand_in.started.wait(5) and stand_in.calls[-1][0][1:] == (two, False)
        stand_in.release(SimpleNamespace(kind="prior", id=two, name="other", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert screen.session.inf_prior.id == two and page.view == "saved"
        assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()
        page.show_view("build")
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try")


def test_a_load_by_code_while_a_run_is_live_lists_nothing_and_starts_nothing(tmp_path, monkeypatch):
    """A load by code asked for while a build is live reads neither list, for nothing is read while a run is live, and
    starts nothing; the build goes on as it was."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        saved = _prior_artifact(default_store(), _nad_cfg(), name="saved_before").id
        page.main_pressed()
        assert stand_in.started.wait(5)
        listed = []
        for name, picker in (("bounds", page.bounds_picker), ("prior", page.prior_picker)):
            monkeypatch.setattr(picker, "refresh",
                                lambda kept=picker.refresh, name=name: (listed.append(name), kept())[1])
        assert page.load_record(saved) is False
        assert listed == [] and len(stand_in.calls) == 1 and controller().is_live(), listed
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)


def test_the_two_lists_are_read_again_at_check_again_and_at_the_show_and_not_as_one_types(tmp_path, monkeypatch):
    """A bounds file put into the model's folder, and a prior written into the store by another program, which tells the
    window nothing, come into the two lists as the page comes into sight again, and at "Check again"; the reading after
    an edit reads neither list. Each list keeps its selection, and what the page says follows the lists: the first
    prior saved ends "No prior is saved yet" as the page is shown again."""
    import shutil
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._look_helpers import _settle
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        folder = tmp_path / "bounds"
        folder.mkdir()
        shutil.copyfile(page.bounds_picker.selected_path(), folder / "master.txt")
        page.bounds_picker.repoint(folder, "master.txt")
        page.show_view("saved")
        page.read_facts()
        files, priors = page.bounds_picker.combo, page.prior_picker.combo

        def listed(combo):
            return [combo.itemData(at) for at in range(combo.count())]

        def said():
            return [row.sentence.plain_text() for row in page.foot.facts.rows()]

        none_saved = "Error. No prior is saved yet: build one with 'Build a new prior instead…'."
        assert none_saved in said(), said()
        shutil.copyfile(folder / "master.txt", folder / "added.txt")
        first = _prior_artifact(default_store(), _nad_cfg(), name="from_the_command_line").id
        page.read_timer.timeout.emit()                   # the reading after an edit
        assert "added.txt" not in listed(files) and first not in listed(priors), "the edit's reading read a list"
        window.go_to("home")
        _settle()
        window.go_to("inference", "prior")
        _settle()
        assert "added.txt" in listed(files) and first in listed(priors), (listed(files), listed(priors))
        assert none_saved not in said() and "Error. Choose a saved prior." in said(), said()
        assert page.bounds_picker.key() == "master.txt" and page.prior_picker.key() == ""
        shutil.copyfile(folder / "master.txt", folder / "another.txt")
        later = _prior_artifact(default_store(), _nad_cfg(), name="written_later").id
        page.foot.check_again.click()
        assert "another.txt" in listed(files) and later in listed(priors), (listed(files), listed(priors))
        assert page.bounds_picker.key() == "master.txt" and page.prior_picker.key() == ""


def test_the_page_shown_again_during_a_run_reads_neither_list_until_it_ends(tmp_path, monkeypatch):
    """While a run is live the page lists nothing as it comes into sight again: left for Home and come back to during a
    build, it reads neither list; the build's end reads both."""
    from core.gui.run.controller import controller
    from tests._look_helpers import _settle
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        page.main_pressed()
        assert stand_in.started.wait(5)
        listed = []
        for name, picker in (("bounds", page.bounds_picker), ("prior", page.prior_picker)):
            monkeypatch.setattr(picker, "refresh",
                                lambda kept=picker.refresh, name=name: (listed.append(name), kept())[1])
        window.go_to("home")
        _settle()
        window.go_to("inference", "prior")
        _settle()
        assert listed == [] and page.isVisible(), listed
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        assert {"bounds", "prior"} <= set(listed), listed


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
    drive can move, are said in "Before you start" as "Attention.", blocking nothing; the scale of the force said of
    what the configuration was built on: the file the list selects, else the file in use by its path, or the limits
    typed by hand, which no file holds."""
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        cfg = screen.session.cfg
        cfg.check_unit_consistency = lambda: ["Units file declares frequency in 'Hz' but time in 'ms'."]
        cfg.rescale_params = {"f_scale": (None, (0.1, 10.0))}
        page.redraw()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows() if row.kind == "warn"]
        why = ("reads a spontaneous recording only, where no drive acts: f_scale cannot change what is observed, so "
               "its estimate will be the prior's. Choose a bounds file with neither a forcing section nor f_scale for "
               "a spontaneous recording.")
        assert said[:2] == [
            "Attention. Units file declares frequency in 'Hz' but time in 'ms'.",
            "Attention. This bounds file lists f_scale, but a posterior trained on it " + why]
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        page.bounds_picker.restore_key("master_tier1.txt")
        page.redraw()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows() if row.kind == "warn"]
        assert said[1] == ("Attention. The bounds file in use, master.txt, lists f_scale, but a posterior trained on "
                           "it " + why), said
        cfg.sources = {"bounds": ""}
        page.redraw()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows() if row.kind == "warn"]
        assert said[1] == ("Attention. The limits typed by hand list f_scale, but a posterior trained on them "
                           + why), said


def test_the_prior_step_says_step_ones_changes_not_applied_and_never_its_own(tmp_path, monkeypatch):
    """A prior's build or load reads the model step 1 holds, never its form: while step 1's page shows a change it has
    not applied, "Before you start" on Prior says so in both views, with the model the run uses, and blocks nothing; a
    change of Prior's own, which its row and the rail say, is no line of its facts."""
    pending = ("Attention. Step 1, Model and options, has changes that are not applied. This run uses what is in use: "
               "Nadrowski.")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        def said():
            return [row.sentence.plain_text() for row in page.foot.facts.rows() if row.kind == "warn"]

        assert pending not in said() and screen.pending() == frozenset(), said()
        screen.config_panel.model_combo.setCurrentText("HOPF")
        assert screen.pending() == {"model"}, screen.pending()
        for view in ("build", "saved"):
            page.show_view(view)
            page.redraw()
            assert said().count(pending) == 1, (view, said())
        page.show_view("build")
        page.redraw()
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        screen.config_panel.model_combo.setCurrentText("NADROWSKI")
        page.bounds_picker.restore_key("master_tier1.txt")
        page.redraw()
        assert screen.pending() == {"prior"}, screen.pending()
        assert not any("has changes that are not applied" in line for line in said()), said()


@pytest.mark.parametrize("end", ["failed", "stopped"])
def test_after_a_load_a_build_that_ends_badly_gives_how_the_run_went_back_at_back_to_the_form(end, tmp_path, monkeypatch):
    """A load that returned its record withdraws "How the run went"; the next build takes the column as it begins, and
    the link is the page's again from then on, however the build ends: one that fails or is stopped keeps its page until
    "Back to the form", which then shows the link to it."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    choose(monkeypatch, words="Build and release the posterior")
    with _second_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        link = page.run_view.how_link
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.release(SimpleNamespace(kind="prior", id="p_new", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        two = _prior_artifact(default_store(), _nad_cfg(), name="other_prior").id
        assert page.load_record(two) is True and stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.release(SimpleNamespace(kind="prior", id=two, name="other_prior", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle_events()
        assert not link.isVisibleTo(page.figures), "the load left the link"
        page.show_view("build")
        page.names.name.setText("retrain_3")
        page.read_facts()
        _press_and_end(page, stand_in, end, monkeypatch)
        _settle_events()
        assert page.results.currentWidget() is page.run_view, f"a build that {end} gave its page up by itself"
        page.run_view.back_button.click()
        _settle_events()
        assert link.isVisibleTo(page.figures), f"after a build that {end}, the link stayed away"


def test_a_bounds_file_edited_since_it_came_into_use_can_be_applied_again(tmp_path, monkeypatch):
    """A prior comes into use on a bounds file, and the file is written again afterwards. Once the page has read it, the
    line under the list says the file in use is the one from before the edit, "Changed, not applied"; the rail hears
    it; and the saved view's button loads the prior again, on the file as it is now, where it said there was nothing to
    apply. A drawing before that reading says nothing of the edit, for the page has not read it. A prior used from
    Records is loaded on the file as it stands, as the button loads it, so a later edit is said of it alike."""
    import os
    import shutil
    from core.artifacts import default_store
    from core.config import BOUNDS_PATH
    from core.gui.panels.inference import prior_tab
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import turn_until
    app = qt_app()
    folder = tmp_path / "Bounds" / "nadrowski"
    folder.mkdir(parents=True)
    path = folder / "master.txt"
    shutil.copyfile(BOUNDS_PATH / "nadrowski" / "master.txt", path)
    monkeypatch.setattr(prior_tab, "BOUNDS_PATH", tmp_path / "Bounds")
    with _second_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        made = _prior_artifact(default_store(), _nad_cfg(), name="on_the_file").id
        page.main_pressed()
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.release(SimpleNamespace(kind="prior", id=made, name="on_the_file", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        page.show_view("saved")
        page.read_facts()
        assert page.prior_picker.key() == made and screen.session.cfg.sources["bounds"] == str(path)
        assert page.foot.reason.text() == "Nothing to apply: these choices are in use."
        assert _line(page, "bounds") == ("", "") and screen.pending() == frozenset()
        held = path.stat()
        path.write_text(path.read_text(encoding="utf-8") + "\n# the file written again\n", encoding="utf-8")
        os.utime(path, ns=(held.st_atime_ns, held.st_mtime_ns + 10 ** 9))
        page.redraw()
        assert _line(page, "bounds") == ("", ""), "a drawing read the file"
        page.read_facts()
        assert _line(page, "bounds") == (
            "warn", "Changed, not applied. In use: master.txt, as it was before it was edited.")
        assert page.foot.main_button.isEnabled() and screen.pending() == frozenset({"prior"})
        candidate = page._candidate
        page.main_pressed()
        assert stand_in.started.wait(5)
        assert stand_in.calls[-1][0][0] is candidate and stand_in.calls[-1][0][1:] == (made, False)
        stand_in.release(SimpleNamespace(kind="prior", id=made, name="on_the_file", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert screen.session.cfg is candidate and _line(page, "bounds") == ("", "")
        assert page.foot.reason.text() == "Nothing to apply: these choices are in use."
        held = path.stat()
        path.write_text(path.read_text(encoding="utf-8") + "# written once more\n", encoding="utf-8")
        os.utime(path, ns=(held.st_atime_ns, held.st_mtime_ns + 10 ** 9))
        assert page.load_record(made)                    # used from Records: loaded on the file as it is now
        assert stand_in.started.wait(5)
        stand_in.started.clear()
        stand_in.release(SimpleNamespace(kind="prior", id=made, name="on_the_file", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        page.read_facts()
        assert _line(page, "bounds") == ("", "")
        held = path.stat()
        path.write_text(path.read_text(encoding="utf-8") + "# and once more\n", encoding="utf-8")
        os.utime(path, ns=(held.st_atime_ns, held.st_mtime_ns + 10 ** 9))
        page.read_facts()
        assert _line(page, "bounds") == (
            "warn", "Changed, not applied. In use: master.txt, as it was before it was edited.")


# ── the third step: a posterior comes into use when its run returns ──
def _held_posterior():
    """A posterior held, as a stand-in that says what a page reads of one."""
    return SimpleNamespace(kind="posterior", id="post_old", name="retrain", manifest=None)


@contextlib.contextmanager
def _third_step(tmp_path, monkeypatch, *, held=True, name="retrain_2"):
    """The third step in a whole window in sight, off the screen, on the test's own clock, its store the test's own: the
    first step applied, Nadrowski with the bounds file's kind; a real prior of the master bounds file in use, with its
    configuration, the Prior page showing that file; with ``held``, a posterior and an observation held too; a name and
    a note typed, the page read, and the stage's training a stand-in run: (window, screen, page, the stand-in run)."""
    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store, use_store
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import StandIn
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window, screen, first = _shown_first_step(tmp_path)
        _apply(first)
        cfg = _nad_cfg()
        made = _prior_artifact(default_store(), cfg, name="retrain_prior").id
        prior = default_store().load_prior(cfg, made)
        session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior)
        if held:
            _cfg, _prior, _posterior, observation = _held_records()
            session.posterior, session.observation = _held_posterior(), observation
        screen.session = session
        screen.prior_panel.bounds_picker.restore_key("master.txt")
        screen.refresh_gates()
        window.go_to("inference", "training")
        page = screen.posterior_panel
        page.names.name.setText(name)
        page.names.note.setText("a second try")
        page.read_facts()
        stand_in = StandIn(word="Training")
        monkeypatch.setattr(orchestrator, "build_posterior", stand_in)
        yield window, screen, page, stand_in


def _start_and_end(page, stand_in, end, monkeypatch, tmp_path):
    """Press "Start training", and end its run as ``end`` names: refused at the stage's entry, failed, stopped in the fit
    with the training it kept then kept or discarded on the run's page, or taken back as it starts."""
    from core.artifacts import default_store
    from core.gui.run.controller import RunController, controller
    from core.refusals import Refusal
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, turn_until
    app = qt_app()
    if end == "withdrawn":
        def broken(_self):
            raise TypeError("the first drawing failed")
        monkeypatch.setattr(RunController, "_draw", broken)
        with pytest.raises(TypeError):
            page._main_clicked()
        return
    text = RunText(controller())
    page._main_clicked()
    assert controller().is_live() and stand_in.started.wait(5)
    if end == "refused before the spend":
        stand_in.fail(Refusal("A posterior named 'retrain_2' was made meanwhile.", field="name"))
    elif end == "failed":
        stand_in.fail(RuntimeError("the fit found no network"))
    else:
        kept = build_fit_record(default_store(), passes=3)
        stand_in.kept_fit = note(kept, passes=3)
        to_the_fit(app, stand_in, text, controller()._clock, tmp_path / "set", ((1, 0), (2, 1), (3, 2)))
        choose(monkeypatch, words="Stop training")
        page.run_view.stop_button.click()
        turn_until(app, lambda: not controller().is_live(), 10.0)
        button = page.run_view.keep_button if end == "stopped, the training kept" else page.run_view.discard_button
        assert button.isVisible(), "the run's page offers neither Keep nor Discard"
        button.click()
        assert [summary.id for summary in default_store().list("fit")] == ([kept] if end.endswith("kept") else [])
        return
    turn_until(app, lambda: not controller().is_live(), 10.0)


@pytest.mark.parametrize("end", ["refused before the spend", "failed", "stopped, the training kept",
                                 "stopped, the training discarded", "withdrawn"])
def test_a_training_that_ends_any_way_but_well_leaves_what_was_held(end, tmp_path, monkeypatch):
    """A training that is refused at the stage's entry, fails, is stopped in its fit and its training kept or discarded on
    the run's page, or is taken back as it starts writes no slot: the configuration, the prior, the posterior and the
    observation held are the very objects they were, the name and the note stand as typed, and the rail, Home's next step
    and "In use" say what they said."""
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        home = window.part("home")
        said = (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("training").accessibleName())
        _start_and_end(page, stand_in, end, monkeypatch, tmp_path)
        assert screen.session is session
        assert all(now is then for now, then in zip((session.cfg, session.inf_prior, session.posterior,
                                                     session.observation), held)), end
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try"), end
        assert (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("training").accessibleName()) == said, end
        assert home is window.part("home")


def test_a_second_press_starts_no_second_training(tmp_path, monkeypatch):
    """Two presses in one turn of the loop start one training: the first greys the button as its run starts, and the
    second finds it so. A third press while the run is live asks the controller nothing, and a press reached with no
    prior held, which no button gives, asks it nothing either."""
    from core.gui.run.controller import RunController, controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        asked = []
        kept = RunController.dispatch
        monkeypatch.setattr(RunController, "dispatch",
                            lambda held, *args, **kwargs: (asked.append(args[0]), kept(held, *args, **kwargs))[1])
        page.foot.main_button.click()
        page.foot.main_button.click()
        assert len(asked) == 1 and controller().is_live() and stand_in.started.wait(5)
        page.foot.main_button.click()
        page._main_clicked()
        page.main_pressed()
        assert len(asked) == 1, "a press while the run was live asked for a second training"
        assert len(stand_in.calls) == 1
        stand_in.release(None)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        screen.session.inf_prior = None
        page.main_pressed()
        assert len(asked) == 1, "a press with no prior held asked for a training"


def test_a_posterior_comes_into_use_only_when_its_run_returns(tmp_path, monkeypatch):
    """The press holds nothing new: the posterior held stays in use through the run. The posterior the run returns comes
    into use while the run is still live, made with the very configuration and prior the press handed the run: what
    depended on the posterior held goes with it, the states are said once, on the window's thread, it is remembered as the
    posterior last in use, and the observation held stays. At the run's end the name and the note are emptied, "In use"
    names the new posterior from its record, and the way to Checks stands."""
    import threading
    from core.gui import settings as st
    from core.gui.remembered import POSTERIOR_KEY
    from core.gui.run import retry
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        session = screen.session
        cfg, prior, old, observation = session.cfg, session.inf_prior, session.posterior, session.observation
        heard, threads = [], []
        screen.states_changed.connect(lambda states: (heard.append(states), threads.append(threading.current_thread())))
        page._main_clicked()
        assert stand_in.started.wait(5)
        args, _kwargs = stand_in.calls[-1]
        assert args[0] is cfg and args[1] is prior and args[2:] == (None, True)
        assert session.posterior is old and controller().is_live()
        assert retry.made_in(controller()._run.spec, session)
        made = SimpleNamespace(kind="posterior", id="20261008T120400", name="retrain_2",
                               manifest=SimpleNamespace(created="2026-10-08T12:04:00+00:00", parents={"prior": prior.id},
                                                        body={"training": {"epochs_trained": 131, "n_runs": 5000,
                                                                           "run_size": 64, "hidden_features": 128,
                                                                           "num_transforms": 8}},
                                                        config={"max_num_epochs": 2147483647}))
        taken = []

        def take(loaded, kept=screen.take_posterior):
            said = len(heard)
            kept(loaded)
            taken.append((controller().is_live(), threading.current_thread(), len(heard) - said))

        monkeypatch.setattr(screen, "take_posterior", take)
        stand_in.release(made)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert taken == [(True, threading.main_thread(), 1)], "the posterior was taken after its run, or more than once"
        assert screen.session is session and session.posterior is made
        assert (session.cfg, session.inf_prior, session.observation) == (cfg, prior, observation)
        assert set(threads) == {threading.main_thread()}
        qs = st.settings()
        qs.sync()
        assert qs.value(POSTERIOR_KEY) == "20261008T120400"
        assert page.in_use.line_text() == "In use: retrain_2"
        assert page.in_use.sentence.plain_text() == (
            "131 epochs, network 128 by 8, on 320,000 simulations, prior retrain_prior. Saved as a record.")
        assert (page.names.name.text(), page.names.note.text()) == ("", "")
        assert not page.next_button.isHidden() and page.next_button.text() == "Go to Checks"


def test_every_argument_of_the_table_reaches_the_stage(tmp_path, monkeypatch):
    """Every setting of the form reaches the stage as an argument of its call, read as the field reads it: the batches, the
    simulations per batch ("Automatic" is 0), the network, the fit's settings, the batches between two saves, the choice
    of continuing, the name and the note, with the call's own words for a taken name and the figures. The epoch limit is
    passed only when one is typed, the rotation's three only while they are read, and the expected epochs never: they
    reach the run's page alone. No constant of the configuration is written."""
    from core import config, orchestrator
    from core.gui.run.controller import RunController
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        seen = []
        monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.append((fn, args, kwargs)))
        constants = {name: getattr(config, name) for name in dir(config) if name.isupper()}
        rows = page.form.rows
        rows["num_runs"].widget.setText("777")
        rows["run_size_cap"].widget.ticked_box.setChecked(False)
        rows["run_size_cap"].widget.field.setText("256")
        rows["hidden_features"].widget.setText("64")
        rows["num_transforms"].widget.setText("3")
        rows["learning_rate"].widget.setText("0.0007")
        rows["stop_after_epochs"].widget.setText("7")
        rows["max_num_epochs"].widget.ticked_box.setChecked(False)
        rows["max_num_epochs"].widget.field.setText("200")
        rows["expected_epochs"].widget.setText("150")
        rows["checkpoint_every"].widget.field.setText("25")
        rows["fisher_m"].widget.setText("12")
        rows["fisher_dz"].widget.setText("0.03")
        rows["fisher_points"].widget.setText("2")
        _pick(page.resume_choice, "never")
        page.read_facts()
        page._main_clicked()
        fn, args, kwargs = seen[-1]
        spec = kwargs.pop("spec")
        assert fn is orchestrator.build_posterior and args == (screen.session.cfg, screen.session.inf_prior, None, True)
        assert {key: value for key, value in kwargs.items() if key not in ("host", "on_result")} == {
            "name": "retrain_2", "note": "a second try", "num_runs": 777, "run_size_cap": 256, "hidden_features": 64,
            "num_transforms": 3, "learning_rate": 0.0007, "stop_after_epochs": 7, "max_num_epochs": 200,
            "checkpoint_every": 25, "resume": "never", "new_run": False, "fisher_m": 12, "fisher_dz": 0.03,
            "fisher_points": 2, "if_name_taken": "unnamed", "provide_fig_sink": True}
        assert kwargs["on_result"] == page._on_posterior and kwargs["host"] is page
        assert (spec.expected_epochs, spec.epoch_limit, spec.patience, spec.name) == (150, 200, 7, "retrain_2")
        rows["max_num_epochs"].widget.ticked_box.setChecked(True)
        rows["run_size_cap"].widget.ticked_box.setChecked(True)
        screen.session.cfg.reparam_rotate = False
        page.read_facts()
        page._main_clicked()
        _fn, _args, kwargs = seen[-1]
        assert "max_num_epochs" not in kwargs and kwargs["run_size_cap"] == 0, kwargs
        assert not {"fisher_m", "fisher_dz", "fisher_points", "expected_epochs"} & set(kwargs), kwargs
        assert {name: getattr(config, name) for name in dir(config) if name.isupper()} == constants


def test_before_you_start_reads_each_input_where_the_page_reads_it(tmp_path, monkeypatch):
    """"Before you start" as the Training page reads each of its inputs, each moved where the page reads it and seen to
    move the fact: the batches, the simulations per batch and the batches between two saves the form holds reach the
    reading of the saved sets, never the usual ones or the remembered ones; the run history's paces are asked for this
    size and this network, and said with their words for the batches still to simulate and the epochs typed; the memory
    limit is said as it stands at each drawing, and the last run on this computer from the run history, with the last
    that measured its batches when the last measured none; a saved set of these settings is continued, the time is for
    the batches still to simulate, and with "Automatic" the paces are asked at the width the reading gives; another
    step's change not applied, step 1's or step 2's, is said with the name of what is in use, as the page is shown
    again, and the line of what the run uses still names the bounds file in use; what the two checks made before a
    training read stands after the memory limit; the name's and every field's facts stand with them. A limit set in the
    environment is set aside, so the window's own setting is said."""
    from datetime import datetime, timedelta, timezone
    from core import config, orchestrator
    from core.gui.memory_limit import computer
    from core.gui.run import history
    from core.gui.run.controller import controller
    from tests._fixtures import write_history
    from tests._look_helpers import _settle
    monkeypatch.delenv("PRISM_VRAM_CEILING_GIB", raising=False)
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        read, paces = [], []
        kept_reading, kept_estimate = orchestrator.training_reading, history.estimate
        monkeypatch.setattr(orchestrator, "training_reading", lambda *args, **kwargs: (
            read.append(kwargs), kept_reading(*args, **kwargs))[1])
        monkeypatch.setattr(history, "estimate", lambda folder, **kwargs: (
            paces.append((kwargs["per"], kwargs["size"])), kept_estimate(folder, **kwargs))[1])
        started = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        write_history(controller().folder(), id="20261008T120000-training", word="Training", name="retrain_1",
                      outcome="finished", started=started, ended=started + timedelta(hours=3),
                      memory_limit={"gib": 6.0, "source": "setting", "computer": computer()},
                      health={"memory_peak": 5.4},
                      simulation_set={"per_batch": 32, "seconds_per_batch": 9.7},
                      fit={"network": [64, 3], "simulations": 777 * 32, "seconds_per_epoch": 60.0})
        rows = page.form.rows
        rows["num_runs"].widget.setText("777")
        rows["run_size_cap"].widget.ticked_box.setChecked(False)
        rows["run_size_cap"].widget.field.setText("32")
        rows["checkpoint_every"].widget.field.setText("25")
        rows["hidden_features"].widget.setText("64")
        rows["num_transforms"].widget.setText("3")
        rows["expected_epochs"].widget.setText("150")
        monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 6.0)
        read.clear()
        paces.clear()
        page.read_facts()
        assert [{key: kwargs[key] for key in ("num_runs", "run_size_cap", "checkpoint_every")} for kwargs in read] == [
            {"num_runs": 777, "run_size_cap": 32, "checkpoint_every": 25}]
        assert paces == [("batch", {"per_batch": 32}), ("epoch", {"network": (64, 3), "simulations": 777 * 32})]

        def said():
            """The facts but the plan's, which its own tests read input by input."""
            return [row.sentence.plain_text() for row in page.foot.facts.rows()
                    if not row.sentence.plain_text().startswith("Information. No plan.")]

        assert said() == [
            "Information. Model Nadrowski; bounds file master.txt; prior retrain_prior; a spontaneous recording and one "
            "driven recording; rotation on.",
            "Information. 777 batches of 32 make 24,864 simulations. Each simulates one spontaneous recording and one "
            "driven recording.",
            "Information. Starts a new simulation set; none is saved for this model yet.",
            "Information. The fit is saved after every epoch, so a stop or a failure keeps it.",
            "Information. A resume first reads the simulation set back, about four minutes at full size.",
            "Information. Simulation: about 2 hours: the last training of this size, retrain_1, took 9.7 s per batch. "
            "Fit: about 2 and a half hours for 150 epochs: the last training of this size, retrain_1, took about 1 "
            "minute per epoch.",
            "Information. Memory limit: 6 GiB for each batch of simulations (set in Settings, This computer). The last "
            "run on this computer (retrain_1, ended 8 Oct 2026, 15:00) ran with a limit of 6 GiB. Its largest batch "
            "held 5.4 GiB.",
            "Information. The two pre-run checks do not apply: they need the frequency response."], said()
        monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 4.0)
        page.redraw()
        assert said()[-2].startswith("Information. Memory limit: 4 GiB for each batch"), said()[-2]
        write_history(controller().folder(), id="20261008T160000-prior-build", word="Prior build", name="later",
                      outcome="finished", started=started + timedelta(hours=4), ended=started + timedelta(hours=5),
                      memory_limit={"gib": 4.0, "source": "setting", "computer": computer()})
        page.read_facts()
        assert said()[-2] == (
            "Information. Memory limit: 4 GiB for each batch of simulations (set in Settings, This computer). The last "
            "run on this computer (later, ended 8 Oct 2026, 17:00) ran with a limit of 4 GiB. The last run that "
            "measured its batches (retrain_1, ended 8 Oct 2026, 15:00): the largest held 5.4 GiB."), said()[-2]
        _a_saved_set(page, 300)                          # 300 of the 777 batches of these settings simulated and saved
        page.read_facts()
        assert said()[2] == "Fine. Continues the saved simulation set: 300 of 777 batches are done.", said()[2]
        assert said()[5].startswith("Information. Simulation: about an hour and a half: the last training of this "
                                    "size, retrain_1, took 9.7 s per batch."), said()[5]
        rows["run_size_cap"].widget.ticked_box.setChecked(True)          # "Automatic": the reading gives the width
        paces.clear()
        page.read_facts()
        width = getattr(page._reading, "width", None)
        assert width and paces == [("batch", {"per_batch": width}),
                                   ("epoch", {"network": (64, 3), "simulations": 777 * width})], (width, paces)
        rows["run_size_cap"].widget.ticked_box.setChecked(False)
        page.read_facts()
        window.go_to("inference", "prior")
        _settle()
        screen.prior_panel.bounds_picker.restore_key("master_tier1.txt")
        window.go_to("inference", "training")
        _settle()
        assert said()[-1] == ("Attention. Step 2, Prior, has changes that are not applied. This run uses what is in "
                              "use: retrain_prior."), said()[-1]
        assert said()[0].startswith("Information. Model Nadrowski; bounds file master.txt; "), said()[0]
        window.go_to("inference", "model")
        _settle()
        kind = screen.config_panel.kind_choice
        kind.setCurrentIndex(kind.findData(not kind.currentData()))
        window.go_to("inference", "training")
        _settle()
        assert said()[-2:] == [
            "Attention. Step 1, Model and options, has changes that are not applied. This run uses what is in use: "
            "Nadrowski.",
            "Attention. Step 2, Prior, has changes that are not applied. This run uses what is in use: "
            "retrain_prior."], said()[-2:]
        rows["learning_rate"].widget.setText("")
        page.names.name.setText("")
        page.redraw()
        assert said()[:2] == ["Error. Type a name.", "Error. Learning rate is blank."], said()[:2]


def test_each_setting_of_a_training_says_its_rule_greys_the_start_and_refuses_the_press_alike(tmp_path, monkeypatch):
    """Each setting a training reads, at the end of its rule: the line under its field says the rule in the row's own
    words, the start is greyed with the words beside it, and the press, made anyway, is refused with the same words, the
    value it found and the usual one; one step inside the rule is no line at all. A blank expected epochs is a choice.
    A note its rule refuses greys the start alike, with the note's words beside it, and the note typed stands."""
    from core.gui.run.controller import RunController
    ends = (("num_runs", "0", "Batches must be at least 1."),
            ("num_runs", "1", None),
            ("run_size_cap", "-1", "Simulations per batch must be at least 0."),
            ("hidden_features", "0", "Network width must be at least 1."),
            ("num_transforms", "0", "Network depth must be at least 1."),
            ("learning_rate", "0", "Learning rate must be greater than 0."),
            ("learning_rate", "1e-9", None),
            ("stop_after_epochs", "0", "End the fit after … epochs without improvement must be at least 1."),
            ("max_num_epochs", "0", "Epoch limit must be at least 1."),
            ("expected_epochs", "0", "Expected epochs (for the time estimate only) must be at least 1."),
            ("expected_epochs", "", None),
            ("checkpoint_every", "-1", "Save the simulations every … batches must be at least 0."),
            ("fisher_m", "0", "Rotation: simulations per perturbation must be at least 1."),
            ("fisher_dz", "0", "Rotation: step must be greater than 0."),
            ("fisher_points", "0", "Rotation: parameter sets averaged must be at least 1."),
            ("hidden_features", "", "Network width is blank."))
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        seen, refused = [], []
        monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.append(kwargs))
        monkeypatch.setattr(page, "_refusal", lambda exc: refused.append(exc))
        rows = page.form.rows
        for key, typed, line in ends:
            widget = rows[key].widget
            field = widget.field if hasattr(widget, "field") else widget
            if hasattr(widget, "ticked_box"):
                widget.ticked_box.setChecked(False)
            usual = field.text()
            field.setText(typed)
            page.redraw()
            assert _line(page, key) == (("err", line) if line else ("", "")), (key, typed)
            assert page.foot.main_button.isEnabled() is (line is None), (key, typed, page.foot.reason.text())
            if line is not None:
                assert page.foot.reason.text() == "Cannot start yet: a setting cannot be used.", key
                page.main_pressed()
                assert refused and refused[-1].field == key and refused[-1].message.startswith(line[:-1]), (
                    key, refused[-1].message if refused else None)
                assert "; got " in refused[-1].message or "is blank" in refused[-1].message
            field.setText(usual)
            if key == "max_num_epochs":
                widget.ticked_box.setChecked(True)
        page.names.note.setText("n" * 201)
        page.redraw()
        assert not page.foot.main_button.isEnabled()
        assert page.foot.reason.text() == "Cannot start yet: the note cannot be used.", page.foot.reason.text()
        page._main_clicked()
        assert page.names.note.text() == "n" * 201
        assert seen == [], "a press with a broken rule started a training"


def test_the_batches_and_the_simulations_per_batch_come_back_kept_from_last_time_until_touched(tmp_path, monkeypatch):
    """The batches and the simulations per batch the last training started with are put back at the next launch as they
    were written, said to be kept from last time until they are touched; a value that is no whole number is passed over
    and the box opens at its usual value."""
    from core import config
    from core.gui.remembered import NUM_RUNS_KEY, RUN_SIZE_KEY
    for remembered, shown, cap in (({NUM_RUNS_KEY: "1234", RUN_SIZE_KEY: "512"}, 1234, 512),
                                   ({NUM_RUNS_KEY: "1234", RUN_SIZE_KEY: "0"}, 1234, 0),
                                   ({NUM_RUNS_KEY: "many", RUN_SIZE_KEY: ""}, config.TRAINING_NUM_RUNS, 0)):
        _remember(**remembered)
        screen = build_inference()
        page = screen.posterior_panel
        rows = page.form.rows
        page.redraw()
        assert (rows["num_runs"].value(), rows["run_size_cap"].value()) == (shown, cap), remembered
        kept = remembered[NUM_RUNS_KEY] == "1234"
        assert (rows["num_runs"].origin.text(), rows["run_size_cap"].origin.text()) == (
            ("kept from last time", "kept from last time") if kept else ("usual", "usual")), remembered
        if kept:
            rows["num_runs"].widget.setText("1235")
            assert (rows["num_runs"].origin.text(), rows["run_size_cap"].origin.text()) == (
                "changed from the usual", "kept from last time")


def test_the_training_page_reads_nothing_while_a_run_is_live(tmp_path, monkeypatch):
    """With a prior in use, so that the page makes every reading it can, nothing it reads is read between a run's start and
    its end, its own training's or another page's: an edit, the timer, "Check again", the press, a store change, the gates
    and the page shown again read nothing; a training that returns its posterior writes the session and reads nothing.
    The run's end reads once: the sets, the history and the paces; and shown again after it, with nothing owed, the page
    reads nothing."""
    from core.gui.run.controller import controller
    from tests._fixtures import readings_of_a_run
    from tests._look_helpers import _settle
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        page._main_clicked()
        assert stand_in.started.wait(5)
        calls = readings_of_a_run(monkeypatch)
        page.form.rows["num_runs"].widget.setText("4321")
        page.read_timer.timeout.emit()
        page.foot.check_again.click()
        page._main_clicked()
        window._store_changed()
        screen.refresh_gates()
        window.go_to("inference", "prior")
        _settle()
        window.go_to("inference", "training")
        _settle()
        assert calls == [], calls
        stand_in.release(SimpleNamespace(kind="posterior", id="p1", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        # the Narrowing round's own lines read the preview, which reads the sets itself: the rest are this page's
        assert (calls.count("training_reading") - calls.count("training_preview"), calls.count("last_on"),
                calls.count("last_measured_on")) == (1, 1, 1), calls
        window.go_to("inference", "model")
        _settle()
        calls.clear()
        window.go_to("inference", "training")
        _settle()
        assert calls == [], f"the page shown again with nothing owed read: {calls}"


def test_the_load_view_reads_nothing_while_a_run_is_live(tmp_path, monkeypatch):
    """The load view reads nothing between a run's start and its end, another page's run: the page shown again with
    nothing owed, another posterior chosen, the timer, "Check again", the press, a store change, the gates and the page
    shown again read no record and no list but the picker's own; a load by code reads nothing, starts nothing and
    changes nothing. The run's end reads the record selected once, with the prior it was trained with and the
    observation it was narrowed around."""
    from core.gui.panels.inference_tabs import InferPanel
    from core.gui.run.controller import controller
    from tests._fixtures import readings_of_a_run
    from tests._look_helpers import _settle
    from tests._runs import running, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        with running(window.panel(InferPanel)):
            calls = readings_of_a_run(monkeypatch)
            window.go_to("inference", "prior")              # the page left and shown again with nothing owed
            _settle()
            window.go_to("inference", "training")
            _settle()
            assert calls == [], calls
            page.post_picker.restore_key(narrowed)
            page.read_timer.timeout.emit()
            page.foot.check_again.click()
            page._main_clicked()
            window._store_changed()
            screen.refresh_gates()
            window.go_to("inference", "prior")
            _settle()
            window.go_to("inference", "training")
            _settle()
            assert page.load_record(trained) is False
            assert calls == [] and page.posterior_line.plain_text().startswith("Made"), calls
            assert (page.view, page.post_picker.key()) == ("load", narrowed)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        # the Narrowing round's own lines read the preview, which reads the sets itself: the rest are this page's, the
        # posterior chosen during the run, the prior it was trained with and the observation it was narrowed around
        assert (calls.count("a record's manifest"), calls.count("training_reading") - calls.count("training_preview"),
                page.post_picker.key()) == (3, 0, narrowed), calls


@pytest.mark.parametrize("step", ["prior", "training"])
def test_a_load_by_code_while_a_run_is_live_changes_nothing(step, tmp_path, monkeypatch):
    """A load by code, as Records will call it, during another page's run: the Prior page and the Training page each
    answer that nothing was started and change nothing, neither the view shown, nor what the list selects, nor what the
    session holds."""
    from core.gui.panels.inference_tabs import InferPanel
    from core.gui.run.controller import controller
    from tests._runs import running, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, _narrowed = _saved_posteriors(page)
        if step == "prior":
            page = screen.prior_panel
            picker, ref, view = page.prior_picker, screen.session.inf_prior.id, "build"
        else:
            picker, ref, view = page.post_picker, trained, "training"
        page.show_view(view)
        picker.restore_key("")
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        with running(window.panel(InferPanel)):
            assert page.load_record(ref) is False
            assert (page.view, picker.key()) == (view, "")
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert (page.view, picker.key()) == (view, "")
        assert screen.session is session and (session.cfg, session.inf_prior, session.posterior,
                                              session.observation) == held


def test_the_training_page_built_during_a_run_says_it_has_read_nothing(tmp_path, monkeypatch):
    """A Training page first built while a run is live reads nothing, says so beside "Before you start", and reads once
    the run has ended."""
    from core.gui.panels.inference.posterior_tab import PosteriorPanel
    from core.gui.panels.inference_tabs import InferPanel
    from core.gui.run.controller import NOT_READ_YET_TEXT, controller
    from tests._fixtures import build_step_page, readings_of_a_run
    from tests._look_helpers import _settle
    from tests._runs import running, turn_until
    app = qt_app()
    window, screen, first = _shown_first_step(tmp_path)
    with running(window.panel(InferPanel)):
        calls = readings_of_a_run(monkeypatch)
        page = build_step_page(PosteriorPanel, window=window)
        window.go_to("inference", "training")
        _settle()
        assert calls == [] and page.foot.reading.text() == NOT_READ_YET_TEXT, calls
    turn_until(app, lambda: not controller().is_live())
    _settle()
    assert page.foot.reading.text() == "The saved records were read at 12:04.", page.foot.reading.text()
    assert (calls.count("last_on"), calls.count("last_measured_on")) == (1, 1), calls


def test_the_run_page_of_a_training_started_on_the_page_keeps_what_was_built_for_it(tmp_path, monkeypatch):
    """A training started by the page's own press has the run's page that a training has: failed during its simulation,
    "Continue from batch N" repeats the page's own call, its name and every setting, with the policy that must continue;
    the graphics card running out of memory, the window's try a minute later repeats it so too; stopped in its fit, the
    run's page asks whether to keep the training and offers to resume it, the call then naming the kept training; and
    both repeat only into the session the page's call was made with."""
    import torch
    from core.artifacts import default_store
    from core.gui.run import retry
    from core.gui.run.controller import controller
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, to_the_simulation, turn_until
    app = qt_app()
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True)
    torch.save({"batches_done": 236, "complete": False}, folder / "state.pt")    # the set as the engine writes it
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        held, view = controller(), page.run_view
        text = RunText(held)
        page._main_clicked()
        assert stand_in.started.wait(5)
        first_args, first_kwargs = stand_in.calls[-1]
        assert retry.made_in(held._run.spec, screen.session)
        to_the_simulation(app, stand_in, text, held, folder, 236)
        stand_in.fail(OSError(28, "No space left on device"))
        turn_until(app, lambda: not held.is_live(), 10.0)
        assert view.continue_button.isVisibleTo(view) and view.continue_button.text() == "Continue from batch 236"
        view.continue_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 2)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "require"})
        assert first_kwargs["name"] == "retrain_2" and first_kwargs["note"] == "a second try"
        lack = RuntimeError("the simulation of batch 237 failed")
        lack.__cause__ = RuntimeError("CUDA error: out of memory")      # the card's error, under the one raised from it
        stand_in.fail(lack)
        turn_until(app, lambda: held.status.state == "retrying", 10.0)
        held._clock.advance(60)
        held.tick()
        turn_until(app, lambda: len(stand_in.calls) == 3 and held.status.state == "running", 10.0)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "require"})
        kept = build_fit_record(default_store(), passes=3)
        stand_in.kept_fit = note(kept, passes=3)
        to_the_fit(app, stand_in, text, held._clock, folder, ((1, 0), (2, 1), (3, 2)), again=False)
        choose(monkeypatch, words="Stop training")
        view.stop_button.click()
        turn_until(app, lambda: not held.is_live(), 10.0)
        assert view.keep_block.isVisibleTo(view) and view.resume_button.isVisibleTo(view)
        view.resume_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 4)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "require", "resume_fit": kept})
        stand_in.release(None)
        turn_until(app, lambda: not held.is_live(), 10.0)


def test_a_training_started_with_only_start_new_is_resumed_on_the_training_it_kept(tmp_path, monkeypatch):
    """"Only start new" guards a training's start, and a kept training is resumed on the simulation set it made, which
    that policy would refuse to read: the window's try after the graphics card runs out of memory in the fit, and
    "Resume the training" after a stop kept with Keep, repeat the page's own call with the kept training named and
    "Continue the saved simulation set if there is one" in its place, and each starts. The stand-in stage refuses, as
    the stage does before anything is spent, a kept training beside "Only start new"."""
    from core import orchestrator
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, turn_until
    app = qt_app()
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True)
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        held, view = controller(), page.run_view
        text = RunText(held)
        refused = []

        def stage(*args, **kwargs):
            if kwargs.get("resume_fit") is not None and kwargs.get("resume") == "never":
                refused.append(kwargs)
                raise Refusal("A kept training is resumed on its kept simulation set, which resume='never' refuses "
                              "to read.", field="resume")
            return stand_in(*args, **kwargs)

        monkeypatch.setattr(orchestrator, "build_posterior", stage)
        _pick(page.resume_choice, "never")
        page.read_facts()
        page._main_clicked()
        assert stand_in.started.wait(5)
        first_args, first_kwargs = stand_in.calls[-1]
        assert first_kwargs["resume"] == "never"
        kept = build_fit_record(default_store(), passes=3)
        stand_in.kept_fit = note(kept, passes=3)
        to_the_fit(app, stand_in, text, held._clock, folder, ((1, 0), (2, 1), (3, 2)))
        lack = RuntimeError("the fit's epoch 4 failed")
        lack.__cause__ = RuntimeError("CUDA error: out of memory")      # the card's error, under the one raised from it
        stand_in.fail(lack)
        turn_until(app, lambda: held.status.state == "retrying", 10.0)
        held._clock.advance(60)
        held.tick()
        turn_until(app, lambda: len(stand_in.calls) == 2 and held.status.state == "running", 10.0)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "auto", "resume_fit": kept})
        to_the_fit(app, stand_in, text, held._clock, folder, ((4, 0),))
        choose(monkeypatch, words="Stop training")
        view.stop_button.click()
        turn_until(app, lambda: not held.is_live(), 10.0)
        assert view.keep_block.isVisibleTo(view)
        view.keep_button.click()
        assert view.resume_button.isVisibleTo(view)
        view.resume_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 3 and held.is_live(), 10.0)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "auto", "resume_fit": kept})
        assert refused == [], "a resume of the kept training was refused"
        stand_in.release(None)
        turn_until(app, lambda: not held.is_live(), 10.0)


# ── step 3, Training ──
def _a_saved_set(page, batches, *, complete=False) -> None:
    """A simulation set in the store in use made with the settings the Training page holds now, as the engine names it:
    the reading at this budget then continues it."""
    from core import orchestrator
    from core.artifacts import default_store
    from tests._fixtures import write_simulation_set
    cfg, prior = page.session.cfg, page.session.inf_prior
    page.read_facts()
    width, runs = page._reading.width, page.form.rows["num_runs"].value()
    write_simulation_set(default_store().kind_dir("simulation"), orchestrator.training_identity(cfg, prior, width, runs),
                         batches, complete=complete)


def _a_set_beside(page, batches, *, complete=True, manifest=False, **changed) -> str:
    """A saved simulation set in the store in use, made with the settings the Training page holds now but those
    ``changed``, entry by entry of its identity, as the engine names it; with ``manifest`` its record too, as the engine
    writes one. Returns its folder's name."""
    from core import orchestrator
    from core.artifacts import default_store, write_simulation_manifest
    from tests._fixtures import write_simulation_set
    cfg, prior = page.session.cfg, page.session.inf_prior
    page.read_facts()
    width, runs = page._reading.width, page.form.rows["num_runs"].value()
    identity = {**orchestrator.training_identity(cfg, prior, width, runs), **changed}
    folder = write_simulation_set(default_store().kind_dir("simulation"), identity, batches, complete=complete)
    if manifest:
        write_simulation_manifest(folder, identity, batches_done=batches, complete=complete)
    return folder.name


def _a_budget(page, runs="777", cap="32") -> None:
    """The batches and the simulations per batch typed, so that a set's identity is the same at every width."""
    rows = page.form.rows
    rows["num_runs"].widget.setText(runs)
    rows["run_size_cap"].widget.ticked_box.setChecked(False)
    rows["run_size_cap"].widget.field.setText(cap)


def _made_words(name) -> str:
    """When the saved set ``name`` was made, as its record says it, in the window's words."""
    from datetime import datetime
    from core.artifacts import default_store
    from core.gui import texts
    return texts.date_text(datetime.fromisoformat(default_store().get("simulation", name).created).astimezone())


# ── step 3's load view, and the kept trainings that match ──
#: How long a narrowed posterior's question is: its words, as the page asks it.
NARROWED_TEXT = ("retrain_round1 was narrowed around the observation cell_b. It is valid only near that observation; "
                 "elsewhere it has seen no training simulations and gives confident answers that are wrong.")
NARROWED_THEN = ("If you load it: calibration runs on the narrowed prior; inference on any other observation asks "
                 "first; "
                 "a further narrowing round can only be drawn around the same observation.")


def _an_observation(name="cell_a") -> str:
    """An observation's record in the store in use, as small as a record of its kind can be: its id."""
    import torch
    from core.artifacts import default_store
    from core.artifacts import manifest as mf
    from core.Helpers import file_manager
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    x_obs = torch.zeros(1, 4, dtype=torch.float64)
    with default_store().create("observation", cfg, name=name) as w:
        file_manager.atomic_torch_save({"x_obs": x_obs, "obs_data": x_obs, "t_dim": x_obs}, w.payload("observation.pt"))
        w.fingerprints["x_obs"] = mf.tensor_digest(x_obs)
        w.body = {"mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
                  "x_obs_digest": mf.tensor_digest(x_obs), "T_obs_cell": 1.0, "n_obs": 4, "forcing_vals": {},
                  "chi_obs_freqs": None, "source": {"kind": "simulated"}}
    return w.id


def _saved_posteriors(page) -> tuple:
    """Two posteriors in the store in use, made for the configuration in use and trained with the prior in use: one a
    training's, "retrain", and one a narrowing round's, "retrain_round1", narrowed around the observation cell_b, which
    is not the observation the session holds: their ids."""
    from core.artifacts import default_store
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    from tests._fixtures import _posterior_artifact
    cfg, prior = page.session.cfg, page.session.inf_prior
    count = len(cfg.params_dict) + len(cfg.rescale_params)
    probe = bijection_probe(reparam.build_inferred_bijection(cfg, log_params=[]), count)
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=count, V=None, probe=probe,
                                       x_obs_digest="d" * 16)
    training = {("training",): {"epochs_trained": 131, "n_runs": 5000, "run_size": 64, "hidden_features": 128,
                                "num_transforms": 8}}
    trained = _posterior_artifact(default_store(), cfg, name="retrain", parents={"prior": prior.id}, over=training).id
    narrowed = _posterior_artifact(default_store(), cfg, name="retrain_round1", amortized=False, region=region,
                                   parents={"prior": prior.id, "observation": _an_observation("cell_b")},
                                   over=training).id
    return trained, narrowed


def _a_posterior_of_unnamed_records() -> tuple:
    """A narrowing round's posterior in the store in use, "of_the_spont_box", made on another bounds file than the one
    in use (the spontaneous box), trained with a prior that has no name and narrowed around an observation that has
    none: its id, and the labels Records gives that prior and that observation."""
    from datetime import datetime
    from core import cli, config, registry
    from core.artifacts import default_store
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    from tests._fixtures import _posterior_artifact, _prior_artifact
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    cfg.hw = config.cpu_device()
    store = default_store()
    prior = _prior_artifact(store, cfg, name="", seed=2).id
    observation = _an_observation(name="")
    count = len(cfg.params_dict) + len(cfg.rescale_params)
    probe = bijection_probe(reparam.build_inferred_bijection(cfg, log_params=[]), count)
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=count, V=None, probe=probe,
                                       x_obs_digest="e" * 16)
    training = {("training",): {"epochs_trained": 131, "n_runs": 5000, "run_size": 64, "hidden_features": 128,
                                "num_transforms": 8}}
    made = _posterior_artifact(store, cfg, name="of_the_spont_box", amortized=False, region=region,
                               parents={"prior": prior, "observation": observation}, over=training).id

    def label(kind, ref):
        return record_words.record_label(kind, "", datetime.fromisoformat(store.get(kind, ref).created).astimezone())

    return made, label("prior", prior), label("observation", observation)


def _to_the_load_view(page) -> None:
    page.show_view("load")
    page.read_facts()


def _the_stage_loads(monkeypatch) -> list:
    """The training stage stood in for by the store's own load of a posterior, with the consent the call hands it: what
    each call was handed, in a list."""
    from core import orchestrator
    from core.artifacts import default_store
    calls = []

    def stage(cfg, prior, ref, train_new, **kwargs):
        calls.append(((cfg, prior, ref, train_new), kwargs))
        return default_store().load_posterior(cfg, ref, accept=kwargs.get("accept"))

    monkeypatch.setattr(orchestrator, "build_posterior", stage)
    return calls


def test_the_saved_posterior_list_says_what_it_holds(tmp_path, monkeypatch):
    """The load view lists the saved posteriors by their labels, a narrowing round's marked in the page's own words and
    nothing offered "from scratch"; under the list stands what the record selected holds, a narrowing round's with the
    observation it was narrowed around. The heading's button goes between the two views, each with its own main button,
    and the training's settings and its block stand in the training's view alone."""
    from datetime import datetime
    from core.artifacts import default_store
    from core.gui import texts
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        assert page.switch_button.text() == "&Use a saved posterior instead…"
        assert page.form.rows["posterior"].is_shown() is False and page.form.rows["num_runs"].is_shown()
        page.switch_button.click()
        page.read_facts()
        assert (page.view, page.switch_button.text(), page.foot.main_button.text()) == (
            "load", "&Set up a training instead…", "&Load this posterior")
        assert page.form.rows["posterior"].is_shown() and not page.form.rows["num_runs"].is_shown()
        assert not page.form.side.isVisibleTo(page) and page.names is None
        combo = page.post_picker.combo
        listed = {combo.itemData(at): combo.itemText(at) for at in range(combo.count())}
        assert listed == {trained: "retrain", narrowed: "retrain_round1  —  from a narrowing round"}, listed
        tips = [combo.itemData(at, 3) for at in range(combo.count())]
        assert not any("TSNPE" in tip or "scratch" in tip for tip in tips), tips
        made = {ref: texts.date_text(datetime.fromisoformat(default_store().get("posterior", ref).created).astimezone())
                for ref in (trained, narrowed)}
        page.post_picker.restore_key(narrowed)
        page.read_facts()
        assert "from a narrowing round" in page.post_picker.selection_summary()
        assert "TSNPE" not in page.post_picker.selection_summary()
        assert page.posterior_line.plain_text() == (
            f"Made {made[narrowed]}, for Nadrowski: a spontaneous recording and one driven recording; network 128 by "
            "8; "
            "320,000 simulations; trained with retrain_prior, on master.txt. From a narrowing round around the "
            "observation cell_b.")
        page.post_picker.restore_key(trained)
        page.read_facts()
        assert page.posterior_line.plain_text().startswith(f"Made {made[trained]}, for Nadrowski: ")
        assert page.posterior_line.plain_text().endswith("on master.txt.")
        other, prior_label, observation_label = _a_posterior_of_unnamed_records()
        page.read_facts()
        page.post_picker.restore_key(other)
        page.read_facts()
        made_other = texts.date_text(
            datetime.fromisoformat(default_store().get("posterior", other).created).astimezone())
        assert page.posterior_line.plain_text() == (
            f"Made {made_other}, for Nadrowski: a spontaneous recording only; network 128 by 8; 320,000 simulations; "
            f"trained with {prior_label}, on master_spont.txt. From a narrowing round around the observation "
            f"{observation_label}."), page.posterior_line.plain_text()
        assert prior_label.startswith("Unnamed prior, ") and observation_label.startswith("Unnamed observation, ")
        page.switch_button.click()
        assert page.view == "training" and page.foot.main_button.text() == "&Start training"
        assert page.form.rows["num_runs"].is_shown() and page.form.side.isVisibleTo(page)


def test_a_saved_posterior_whose_moment_cannot_be_read_is_said_without_it(tmp_path, monkeypatch):
    """A saved posterior whose record holds a moment that cannot be read as one is said from its model on, with no
    "Made" and no empty moment."""
    import json
    from core.artifacts import default_store
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, _narrowed = _saved_posteriors(page)
        path = next(summary.path for summary in default_store().list("posterior") if summary.id == trained)
        record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        record["created"] = "the third of October"
        (path / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        assert page.posterior_line.plain_text().startswith("For Nadrowski: a spontaneous recording and one driven "
                                                          "recording; network 128 by 8;"), (
            page.posterior_line.plain_text())


def test_a_narrowed_posterior_asks_before_it_loads_and_a_consent_is_never_kept(tmp_path, monkeypatch):
    """A posterior from a narrowing round asks first, in its words, "Cancel" preselected: "Cancel" starts nothing and
    leaves the session as it was; "Load this posterior" loads it with the consent, its call the load's, and the region
    it was narrowed in comes into the session with it. The consent is kept nowhere: a second load asks again, and the
    settings file holds no word of it. A training's posterior is loaded with no question and no consent. A narrowed
    record the page could not read as narrowed is refused by the store for want of the consent, and the refusal says
    where the consent is given."""
    from core.artifacts import Accept, default_store
    from core.gui import settings as st
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        calls = _the_stage_loads(monkeypatch)
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        _to_the_load_view(page)
        page.post_picker.restore_key(narrowed)
        page.read_facts()
        page._main_clicked()
        question = asked()
        assert (question.title, question.sentence, question.beneath, question.answers, question.default) == (
            "Load a posterior from a narrowing round?", NARROWED_TEXT, NARROWED_THEN,
            ("Load this posterior", "Cancel"), "Cancel")
        assert calls == [] and (session.cfg, session.inf_prior, session.posterior, session.observation) == held
        choose(monkeypatch, words="Load this posterior")
        page._main_clicked()
        turn_until(app, lambda: not controller().is_live() and calls, 10.0)
        (args, kwargs), = calls
        assert args == (held[0], held[1], narrowed, False) and kwargs["accept"] == Accept(truncated=True)
        assert session.posterior.id == narrowed and session.posterior.posterior.truncation is not None
        forget_asked()
        choose(monkeypatch, words="Cancel")
        page.post_picker.restore_key(trained)
        page.read_facts()
        page._main_clicked()
        turn_until(app, lambda: not controller().is_live() and len(calls) == 2, 10.0)
        assert all_asked() == [] and calls[-1][1]["accept"] == Accept() and session.posterior.id == trained
        page.post_picker.restore_key(narrowed)
        page.read_facts()
        page._main_clicked()
        assert asked().title == "Load a posterior from a narrowing round?" and len(calls) == 2, "the consent was kept"
        held_keys = st.settings().allKeys()
        assert not [key for key in held_keys if "accept" in key.lower() or "truncat" in key.lower()], held_keys
        page._record["narrowed"] = False                 # as a record rewritten since the page read it would read
        page.main_pressed()
        turn_until(app, lambda: not controller().is_live() and len(calls) == 3, 10.0)
        refusal = asked()
        assert refusal.title == "Check your inputs" and refusal.beneath == (
            "Answer 'Load this posterior' in the question that loading a posterior from a narrowing round asks on this "
            "page."), refusal
        assert session.posterior.id == trained


def test_the_load_views_facts_read_the_record_selected(tmp_path, monkeypatch):
    """The load view's "Before you start" reads the record selected, not what the session holds: a posterior trained
    with another prior says which, warns that calibration and a narrowing round will refuse the pair, and still loads;
    one made for another model blocks the load and names that model; one whose record cannot be read blocks it and says
    why."""
    from core.artifacts import ArtifactStore, default_store
    from tests._fixtures import _nad_cfg, _posterior_artifact, _prior_artifact
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        cfg, held = page.session.cfg, page.session.inf_prior.id
        store = default_store()
        other = _prior_artifact(store, _nad_cfg(), name="other_prior", seed=1).id
        with_other = _posterior_artifact(store, cfg, name="with_other", parents={"prior": other}).id
        for_hopf = _posterior_artifact(store, cfg, name="for_hopf", parents={"prior": held},
                                       over={("config", "model"): "HOPF"}).id
        unread = _posterior_artifact(store, cfg, name="unread", parents={"prior": held}).id
        real = ArtifactStore.get

        def get(self, kind, ref):
            if ref == unread:
                raise OSError("the disk said no")
            return real(self, kind, ref)

        monkeypatch.setattr(ArtifactStore, "get", get)
        _to_the_load_view(page)
        seen = {}
        for ref in (with_other, for_hopf, unread):
            page.post_picker.restore_key(ref)
            page.read_facts()
            seen[ref] = ([row.sentence.plain_text() for row in page.foot.facts.rows()],
                         page.foot.main_button.isEnabled(), page.foot.reason.text())
        assert seen[with_other] == (
            ["Attention. Trained with other_prior; the prior in use is retrain_prior. Calibration and a narrowing "
             "round "
             "will refuse this pair. Load other_prior in step 2 first.", NO_PLAN], True, ""), seen[with_other]
        assert seen[for_hopf] == (["Error. This posterior was made for the model Hopf.", NO_PLAN], False,
                                  "Cannot start: this posterior is for another model."), seen[for_hopf]
        assert seen[unread] == (["Error. The posterior cannot be read: OSError: the disk said no", NO_PLAN], False,
                                "Cannot start: the posterior cannot be read."), seen[unread]


def test_a_load_reads_none_of_the_training_settings(tmp_path, monkeypatch):
    """A load reads none of the training's settings: with the batches blank and the learning rate below its rule, the
    load view is not greyed, the load starts, and its call holds the posterior, the consent and nothing of the form."""
    from core import orchestrator
    from core.gui.run.controller import RunController
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, _narrowed = _saved_posteriors(page)
        page.form.rows["num_runs"].widget.setText("")
        page.form.rows["learning_rate"].widget.setText("-1")
        page.read_facts()
        assert not page.foot.main_button.isEnabled()
        seen = []
        monkeypatch.setattr(RunController, "dispatch",
                            lambda held, fn, *args, **kwargs: seen.append((fn, args, kwargs)))
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        page._main_clicked()
        fn, args, kwargs = seen[-1]
        spec = kwargs.pop("spec")
        assert fn is orchestrator.build_posterior
        assert args == (screen.session.cfg, screen.session.inf_prior, trained, False)
        assert sorted(kwargs) == ["accept", "host", "on_result", "provide_fig_sink"], sorted(kwargs)
        assert kwargs["provide_fig_sink"] is False and spec.loads_only and spec.word == "Posterior load"


def _press_the_load_and_end(page, stand_in, end, monkeypatch):
    """Press "Load this posterior", and end its run as ``end`` names: refused at the stage's entry, failed, stopped, or
    taken back as it starts."""
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
        stand_in.fail(Refusal("Posterior 'retrain' was trained for model HOPF.", field="posterior"))
    elif end == "failed":
        stand_in.fail(RuntimeError("the posterior's file could not be read"))
    else:
        controller().request_stop()
    turn_until(app, lambda: not controller().is_live(), 10.0)


def test_a_load_by_code_selects_reads_and_asks_as_the_press_does(tmp_path, monkeypatch):
    """A load by code, the chain of loads' link, shows the load view, reads the list so that a posterior saved since by
    another program is found, selects it and reads its record, then loads it as "Load this posterior" does: a training's
    with no question, a narrowing round's asked first, "Cancel" starting nothing; a posterior no longer listed, or one
    the facts block, starts nothing."""
    from core.artifacts import Accept
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        calls = _the_stage_loads(monkeypatch)
        trained, narrowed = _saved_posteriors(page)          # saved after the page last read its list
        assert page.view == "training"
        assert page.load_record(trained) is True
        turn_until(app, lambda: not controller().is_live() and calls, 10.0)
        assert all_asked() == [] and calls[-1][0][2] == trained and calls[-1][1]["accept"] == Accept()
        assert page.view == "load" and page.post_picker.key() == trained
        assert page.load_record(narrowed) is False, "Cancel started the load"
        assert asked().title == "Load a posterior from a narrowing round?" and len(calls) == 1
        choose(monkeypatch, words="Load this posterior")
        assert page.load_record(narrowed) is True
        turn_until(app, lambda: not controller().is_live() and len(calls) == 2, 10.0)
        assert calls[-1][1]["accept"] == Accept(truncated=True)
        assert page.load_record("20200101T000000") is False and len(calls) == 2
        screen.session.inf_prior = None
        assert page.load_record(trained) is False and len(calls) == 2


@pytest.mark.parametrize("end", ["refused before the spend", "failed", "stopped", "withdrawn"])
def test_a_load_that_ends_any_way_but_well_leaves_the_posterior_held(end, tmp_path, monkeypatch):
    """A load that is refused at the stage's entry, fails, is stopped or is taken back as it starts writes no slot: the
    configuration, the prior, the posterior and the observation held are the very objects they were, the rail, Home
    and "In use" say what they said, and the posterior last in use is still the one remembered."""
    from core.gui import settings as st
    from core.gui.remembered import POSTERIOR_KEY, put
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        put(POSTERIOR_KEY, "post_old")
        trained, _narrowed = _saved_posteriors(page)
        screen.refresh_gates()                           # the observation the records were written beside is listed
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        session = screen.session
        held = (session.cfg, session.inf_prior, session.posterior, session.observation)
        home = window.part("home")
        said = (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("training").accessibleName())
        stand_in.word = "Posterior load"
        _press_the_load_and_end(page, stand_in, end, monkeypatch)
        assert screen.session is session
        assert all(now is then for now, then in zip((session.cfg, session.inf_prior, session.posterior,
                                                     session.observation), held)), end
        assert (screen.step_states(), page.in_use.line_text(), page.in_use.sentence.plain_text(),
                window.shell.rail.row("training").accessibleName()) == said, end
        assert home is window.part("home") and page.results.currentWidget() is page.figures, end
        qs = st.settings()
        qs.sync()
        assert qs.value(POSTERIOR_KEY) == "post_old", end
        assert page.post_picker.key() == trained, end


def test_loading_a_posterior_while_one_is_held_asks_nothing(tmp_path, monkeypatch):
    """With a posterior held, a training's posterior is loaded with no question: the posterior held stays in use through
    the load, the loaded one comes into use only as its run returns it, is remembered, and the list is moved to it at
    the run's end; "In use" names it from its record, and the way to Checks stands. The observation held stays."""
    from core.gui import settings as st
    from core.gui.remembered import POSTERIOR_KEY
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        session = screen.session
        old, observation = session.posterior, session.observation
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        listed, refresh = [], page.post_picker.refresh
        monkeypatch.setattr(page.post_picker, "refresh", lambda: (listed.append(controller().is_live()), refresh())[1])
        page._main_clicked()
        assert all_asked() == [] and stand_in.started.wait(5)
        assert session.posterior is old and page.results.currentWidget() is page.figures
        assert not page.switch_button.isEnabled(), "the way to the other view stood open during the load"
        page.post_picker.restore_key(narrowed)            # a choice moved during the load is not what it loaded
        from core.artifacts import default_store
        loaded = default_store().load_posterior(session.cfg, trained)
        stand_in.release(loaded)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert session.posterior is loaded and session.observation is observation
        assert page.post_picker.key() == trained, "the list was not moved to the posterior loaded"
        assert listed and not any(listed), ("the list was read while the load was live", listed)
        qs = st.settings()
        qs.sync()
        assert qs.value(POSTERIOR_KEY) == trained
        assert page.in_use.line_text() == "In use: retrain"
        assert page.in_use.sentence.plain_text() == (
            "131 epochs, network 128 by 8, on 320,000 simulations, prior retrain_prior. Saved as a record.")
        assert not page.next_button.isHidden() and page.next_button.text() == "Go to Checks"
        assert page.foot.reason.text() == "Nothing to apply: these choices are in use."
        page.post_picker.restore_key(narrowed)
        page.read_facts()
        assert _line(page, "posterior") == ("warn", "Changed, not applied. In use: retrain.")


def test_the_stop_and_end_of_a_posterior_load_say_what_is_true_of_the_page(tmp_path, monkeypatch):
    """A load takes no run's page: its stop asks nothing and keeps nothing, the form stays in its place, and the
    posterior held stays in use; a load finished shows the record loaded, "How the run went" away, and says it in "In
    use", a narrowing round's said to be one."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        held = screen.session.posterior
        _to_the_load_view(page)
        page.post_picker.restore_key(narrowed)
        page.read_facts()
        choose(monkeypatch, words="Load this posterior")
        stand_in.word = "Posterior load"
        page._main_clicked()
        assert stand_in.started.wait(5) and page.results.currentWidget() is page.figures
        forget_asked()
        controller().request_stop()
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert all_asked() == [] and screen.session.posterior is held
        assert page.results.currentWidget() is page.figures
        page._main_clicked()
        from core.artifacts import Accept, default_store
        loaded = default_store().load_posterior(screen.session.cfg, narrowed, accept=Accept(truncated=True))
        turn_until(app, lambda: controller().is_live() and len(stand_in.calls) == 2, 10.0)
        stand_in.release(loaded)
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert page.results.currentWidget() is page.figures and not page.run_view.how_link.isVisibleTo(page.figures)
        assert page.in_use.line_text() == "In use: retrain_round1"
        assert page.in_use.sentence.plain_text() == (
            "131 epochs, network 128 by 8, on 320,000 simulations, prior retrain_prior. From a narrowing round. Saved "
            "as a record.")


def _a_narrowing_round_brings(window, screen, narrowed, monkeypatch, *, heard=None):
    """Run the narrowing round of step 6 from its foot's button, its stage a stand-in that returns the saved posterior
    ``narrowed``, loaded as the round's stage hands its posterior back, and turn until its run has ended; ``heard`` is
    called each time the steps' states are worked out again while the run goes. The posterior returned."""
    from core import orchestrator
    from core.artifacts import Accept, default_store
    from core.gui.panels.inference import tsnpe_tab
    from core.gui.run.controller import controller
    from tests._fixtures import write_observation
    from tests._runs import StandIn, turn_until
    app = qt_app()
    loaded = default_store().load_posterior(screen.session.cfg, narrowed, accept=Accept(truncated=True))
    rounds = StandIn(word="Narrowing round")
    monkeypatch.setattr(orchestrator, "tsnpe_round", rounds)
    observation = SimpleNamespace(kind="observation", id="obs_round", name="cell_b")
    monkeypatch.setattr(tsnpe_tab, "default_store",
                        lambda: SimpleNamespace(load_observation=lambda cfg, ref: observation))
    round_page = screen.tsnpe_panel
    observed = write_observation(default_store(), screen.session.cfg)
    window.go_to("inference", "narrowing")
    screen.refresh_gates()
    round_page.read_facts()
    round_page.obs_picker.restore_key(observed)
    round_page.names.name.setText("retrain_round2")
    round_page.read_facts()
    assert round_page.obs_picker.key() and round_page.foot.main_button.isEnabled(), round_page.foot.reason.text()
    round_page.foot.main_button.click()
    assert rounds.started.wait(5)
    if heard is not None:
        screen.states_changed.connect(heard)
    try:
        rounds.release(loaded)
        turn_until(app, lambda: not controller().is_live(), 10.0)
    finally:
        if heard is not None:
            screen.states_changed.disconnect(heard)
    return loaded


def test_a_narrowed_posterior_in_use_says_it_came_from_a_narrowing_round_wherever_it_came_from(tmp_path, monkeypatch):
    """A posterior from a narrowing round in use, put there by the narrowing round's own run, says so in step 3's "In
    use" as one loaded on step 3 does."""
    with _third_step(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        _trained, narrowed = _saved_posteriors(page)
        _a_narrowing_round_brings(window, screen, narrowed, monkeypatch)
        assert screen.session.posterior.id == narrowed
        assert page.in_use.sentence.plain_text().endswith("From a narrowing round. Saved as a record.")


@pytest.mark.parametrize("view", ["training", "load"])
def test_a_posterior_a_narrowing_round_brings_moves_the_list_and_leaves_nothing_to_apply(view, tmp_path, monkeypatch):
    """A posterior a narrowing round puts in use, by its own run, is what step 3 holds from then on: once the run has
    ended the list of saved posteriors stands on it, whichever view the Training page was in, so the rail, Home and the
    page say step 3 holds it with nothing changed, and "Load this posterior" is not offered for the posterior it
    replaced. It is remembered as the posterior last in use. While the run is still live nothing is read, and the list
    is not moved. A posterior chosen in the list after that is the person's choice, which the end of another run
    leaves as it is."""
    from core.gui import settings as st
    from core.gui.panels.inference_tabs import InferPanel
    from core.gui.remembered import POSTERIOR_KEY
    from core.gui.run.controller import controller
    from tests._look_helpers import _settle
    from tests._runs import running, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, narrowed = _saved_posteriors(page)
        calls = _the_stage_loads(monkeypatch)
        _to_the_load_view(page)
        page.post_picker.restore_key(trained)
        page.read_facts()
        page._main_clicked()
        turn_until(app, lambda: not controller().is_live() and calls, 10.0)
        assert screen.session.posterior.id == trained and page.post_picker.key() == trained
        page.show_view(view)
        heard = []
        _a_narrowing_round_brings(window, screen, narrowed, monkeypatch, heard=lambda states: heard.append(
            (controller().is_live(), getattr(screen.session.posterior, "id", None), page.post_picker.key(),
             states[2].pending)))
        while_live = [key for live, _held, key, _pending in heard if live]
        assert while_live and set(while_live) == {trained}, heard
        window.go_to("inference", "training")
        _settle()
        if view == "training":
            page.show_view("load")
        page.read_facts()
        assert screen.session.posterior.id == narrowed and page.post_picker.key() == narrowed
        state = screen.step_states()[2]
        assert (state.pending, state.rail_words) == (False, "Done: retrain_round1, narrowed"), state
        assert screen.pending() == frozenset()
        assert "not applied" not in window.shell.rail.row("training").accessibleName()
        assert _line(page, "posterior") == ("", "")
        assert not page.foot.main_button.isEnabled()
        assert page.foot.reason.text() == "Nothing to apply: these choices are in use."
        qs = st.settings()
        qs.sync()
        assert qs.value(POSTERIOR_KEY) == narrowed
        page.post_picker.restore_key(trained)
        page.read_facts()
        with running(window.panel(InferPanel)):
            pass
        assert page.post_picker.key() == trained and screen.pending() == {"training"}, "the end of a run undid a choice"


def test_a_posterior_saved_while_the_page_is_hidden_is_listed_when_it_is_shown(tmp_path, monkeypatch):
    """The load view shown again with nothing owed lists the saved posteriors again, so that one saved meanwhile by
    another program, of which the window heard nothing, is offered."""
    from tests._fixtures import _posterior_artifact
    from tests._look_helpers import _settle
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, _narrowed = _saved_posteriors(page)
        _to_the_load_view(page)
        window.go_to("inference", "prior")
        _settle()
        from core.artifacts import default_store
        later = _posterior_artifact(default_store(), page.session.cfg, name="saved_meanwhile").id
        assert not page.stale and page.post_picker.combo.findData(later) < 0
        window.go_to("inference", "training")
        _settle()
        assert page.post_picker.combo.findData(later) >= 0, "a posterior saved meanwhile was not listed"


@pytest.mark.parametrize("gone", [False, True], ids=["still listed", "no longer listed"])
def test_the_posterior_last_in_use_comes_back_as_the_lists_choice(gone, tmp_path, monkeypatch):
    """The posterior that came into use last is put back at the next launch as the choice of the list of saved
    posteriors, never loaded, and said kept from last time until it is touched; one no longer listed selects nothing and
    is not said to be kept."""
    import shutil
    from core.artifacts import default_store
    from core.gui.remembered import POSTERIOR_KEY, put
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        trained, _narrowed = _saved_posteriors(page)
        put(POSTERIOR_KEY, trained)
        if gone:
            shutil.rmtree(next(summary.path for summary in default_store().list("posterior") if summary.id == trained))
        from core.gui.panels.inference.posterior_tab import PosteriorPanel
        from tests._fixtures import build_part
        again = build_part(PosteriorPanel, screen=screen)
        assert again.post_picker.key() == ("" if gone else trained)
        assert again.form.rows["posterior"].kept is (not gone)
        assert screen.session.posterior is None
        again.show_view("load")
        again.redraw()
        assert again.form.rows["posterior"].origin.text() == ("" if gone else "kept from last time")
        if not gone:
            again.post_picker.restore_key(_narrowed)
            assert again.form.rows["posterior"].kept is False


def _a_kept_training(page, *, stopped="on_purpose", name="retrain_1", passes=41) -> str:
    """A kept training in the store in use that the Training page's settings now match, as the stage would name it: its
    simulation set, of these settings and whole, beside it; its last pass saved an hour after it was first kept. Its
    id."""
    from core import orchestrator
    from core.artifacts import default_store
    from core.SBI import fit_state
    from tests._fixtures import build_fit_record, write_simulation_set
    from datetime import datetime, timedelta, timezone
    cfg, prior = page.session.cfg, page.session.inf_prior
    page.read_facts()
    width, runs = page._reading.width, page.form.rows["num_runs"].value()
    folder = write_simulation_set(default_store().kind_dir("simulation"),
                                  orchestrator.training_identity(cfg, prior, width, runs), runs, complete=True)
    named, kept = [], fit_state.matching_kept_fits
    fit_state.matching_kept_fits = lambda _store, identity: named.append(identity) or []
    try:
        values = page._stage_values()
        orchestrator.matching_fits(cfg, prior, num_runs=runs, run_size_cap=values["run_size_cap"],
                                   checkpoint_every=values["checkpoint_every"],
                                   **{key: values.get(key) for key in ("hidden_features", "num_transforms",
                                                                       "learning_rate", "stop_after_epochs",
                                                                       "max_num_epochs")})
    finally:
        fit_state.matching_kept_fits = kept
    identity, = named
    return build_fit_record(default_store(), passes=passes, stopped=stopped, posterior_name=name,
                            identity={**identity, "network": "0" * 16}, parents={"simulation": folder.name},
                            saved_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())


def _saved_words(ref) -> str:
    """When the kept training ``ref`` was last saved, in the window's words, as its record says it."""
    from datetime import datetime
    from core.artifacts import default_store
    from core.gui import texts
    saved = default_store().get("fit", ref).body["progress"]["saved_at"]
    return texts.date_text(datetime.fromisoformat(saved).astimezone())


def test_a_kept_training_that_matches_is_named_before_a_new_one_starts(tmp_path, monkeypatch):
    """Kept trainings that match the training about to start, at the network, the learning rate, the patience and the
    epoch limit typed, each off its usual value, are named first, in a question, "Cancel" preselected: each by its name
    or its label, how it stopped in words, the epoch it was saved after and when that pass was saved, in local time, not
    when it was first kept. "Cancel" starts nothing; "Start a new training" starts one beside them, which leaves each
    kept training as it is, and asks nothing more: a match is of a whole simulation set, which a training continues."""
    from datetime import datetime
    from core.artifacts import default_store
    from core.gui import texts
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page, runs="12", cap="8")
        rows = page.form.rows
        rows["hidden_features"].widget.setText("96")
        rows["learning_rate"].widget.setText("0.00025")
        rows["stop_after_epochs"].widget.setText("7")
        rows["max_num_epochs"].widget.ticked_box.setChecked(False)
        rows["max_num_epochs"].widget.field.setText("200")
        assert all(rows[key].value() != rows[key].usual()
                   for key in ("hidden_features", "learning_rate", "stop_after_epochs", "max_num_epochs"))
        first = _a_kept_training(page)
        second = _a_kept_training(page, stopped="error", name="", passes=7)
        page.read_facts()
        assert [kept.id for kept in page._matches] == [second, first]
        page._main_clicked()
        question = asked()
        made = datetime.fromisoformat(default_store().get("fit", second).created).astimezone()
        unnamed = record_words.record_label("fit", "", made)
        assert (question.title, question.sentence, question.answers, question.default) == (
            "Start a new training?", "2 kept trainings match these settings and can be resumed:",
            ("Start a new training", "Cancel"), "Cancel")
        assert question.items == (
            f"{unnamed}: stopped unexpectedly after the error 'RuntimeError: the card ran out of memory', after "
            "epoch 7, "
            f"saved {_saved_words(second)}",
            f"retrain_1: kept when stopped, after epoch 41, saved {_saved_words(first)}"), question.items
        assert question.beneath == ("A new training leaves them as they are. To resume one, cancel and press Resume "
                                    "beside it on this page.")
        assert stand_in.calls == []
        choose(monkeypatch, words="Start a new training")
        forget_asked()
        page._main_clicked()
        _the_call(stand_in, 1)
        assert [said.title for said in all_asked()] == ["Start a new training?"]
        assert stand_in.calls[-1][1]["new_run"] is False
        assert (stand_in.calls[-1][1]["hidden_features"], stand_in.calls[-1][1]["learning_rate"]) == (96, 0.00025)
        assert _saved_words(first) != texts.date_text(
            datetime.fromisoformat(default_store().get("fit", first).created).astimezone())
        stand_in.release(None)
        _settle_the_run()
        assert sorted(summary.id for summary in default_store().list("fit")) == sorted([first, second])


@pytest.mark.parametrize("match, beside", [(False, False), (True, False), (False, True), (True, True)],
                         ids=["neither", "a kept training", "a set beside", "both"])
def test_at_most_one_of_the_two_questions_is_asked(match, beside, tmp_path, monkeypatch):
    """A press asks at most one question before a training: a kept training that matches is named, and its answer starts
    the training; else a set of the model saved beside asks before simulating from zero; with neither, nothing."""
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page, runs="12", cap="8")
        if beside:
            _a_set_beside(page, 40, n_runs=40)
        if match:
            _a_kept_training(page)
        page.read_facts()
        choose(monkeypatch, other=True)
        page._main_clicked()
        _the_call(stand_in, 1)
        titles = [said.title for said in all_asked()]
        wanted = ["Start a new training?"] if match else ["Start a new simulation set?"] if beside else []
        assert titles == wanted, titles
        assert stand_in.calls[-1][1]["new_run"] is (beside and not match)
        stand_in.release(None)
        _settle_the_run()


def test_a_refusal_after_a_match_reads_the_list_again(tmp_path, monkeypatch):
    """A training started beside a kept training that matches can still be refused by the stage before anything is
    spent: its run's end reads the kept trainings again, so one deleted meanwhile is named no more at the next press."""
    import shutil
    from core.artifacts import default_store
    from core.refusals import Refusal
    from tests._fixtures import readings_of_a_run
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page, runs="12", cap="8")
        kept = _a_kept_training(page)
        page.read_facts()
        choose(monkeypatch, words="Start a new training")
        page._main_clicked()
        _the_call(stand_in, 1)
        calls = readings_of_a_run(monkeypatch)
        shutil.rmtree(next(summary.path for summary in default_store().list("fit") if summary.id == kept))
        stand_in.fail(Refusal("A posterior named 'retrain_2' was made meanwhile.", field="name"))
        _settle_the_run()
        assert calls.count("matching_fits") == 1, calls
        assert page._matches == ()
        forget_asked()
        page.names.name.setText("retrain_3")
        page.read_facts()
        page._main_clicked()
        _the_call(stand_in, 2)
        assert all_asked() == []
        stand_in.release(None)
        _settle_the_run()


@pytest.mark.parametrize("which", ["a kept training", "from zero", "a narrowed posterior"])
def test_both_questions_of_the_training_page_preselect_cancel(which, tmp_path, monkeypatch):
    """Every question the Training page asks before it spends, the kept training's, the one before simulating from zero
    and a narrowed posterior's, preselects "Cancel", the safe answer: answered as Enter answers it, nothing starts."""
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page, runs="12", cap="8")
        if which == "a kept training":
            _a_kept_training(page)
        elif which == "from zero":
            _a_set_beside(page, 40, n_runs=40)
        else:
            _trained, narrowed = _saved_posteriors(page)
            _to_the_load_view(page)
            page.post_picker.restore_key(narrowed)
        page.read_facts()
        choose(monkeypatch, default=True)
        page._main_clicked()
        question = asked()
        assert (question.default, question.safe, question.answers[-1]) == ("Cancel", "Cancel", "Cancel"), question
        assert stand_in.calls == []


def test_a_saved_set_of_this_model_asks_first_and_lists_every_difference(tmp_path, monkeypatch):
    """A training that would simulate from zero while saved sets of its model exist asks first, "Cancel" preselected:
    the batches it will simulate, then each set by its label in Records with when it was made where its record says,
    what it holds and whether it is whole, the most batches first, each followed by every setting it differs in by its
    name on screen with both values, a set made with another prior naming that prior by its label. A set of another
    model is not listed. A set whose limits differ names each limit by the parameter of the bounds file in use; a prior
    with no name is named by its label in Records. "Cancel" starts nothing; the other answer starts the training with
    the consent to start a new set and every other argument as the press reads it. A reading that cannot read the saved
    sets says so among the facts, blocks nothing and asks nothing, and the training then starts without the consent."""
    from datetime import datetime
    from core import orchestrator
    from core.artifacts import default_store
    from core.SBI import training_checkpoint
    from tests._fixtures import _nad_cfg, _prior_artifact
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page)
        rotate = bool(screen.session.cfg.reparam_rotate)
        unnamed = _prior_artifact(default_store(), _nad_cfg(), name="", seed=1)
        other = unnamed.fingerprints["gmm"]
        other_label = record_words.record_label(
            "prior", "", datetime.fromisoformat(default_store().get("prior", unnamed.id).created).astimezone())
        whole = _a_set_beside(page, 4000, manifest=True, n_runs=4000, run_size=1024, reparam_rotate=not rotate)
        _a_set_beside(page, 12, complete=False, prior_fingerprint=other)
        _a_set_beside(page, 900, model="HOPF")
        cfg, prior = page.session.cfg, page.session.inf_prior
        ident = orchestrator.training_identity(cfg, prior, page._reading.width, 777)
        lows, highs = list(ident["nd_lows"]), list(ident["rescale_highs"])
        mine = (lows[0], highs[-1])
        lows[0], highs[-1] = 0.75, 99.5
        _a_set_beside(page, 30, nd_lows=lows, rescale_highs=highs)
        first, last = list(cfg.params_dict)[0], list(cfg.rescale_params)[-1]
        page.read_facts()
        page._main_clicked()
        question = asked()
        assert (question.title, question.answers, question.default) == (
            "Start a new simulation set?", ("Start training on a new simulation set", "Cancel"), "Cancel")
        assert question.sentence == ("This training will simulate 777 batches from zero. Saved simulation sets for "
                                     "this "
                                     "model exist, and none matches these settings.")
        words = {True: "on", False: "off"}
        assert question.items == (
            f"Simulation set for retrain_prior ({_made_words(whole)}; 4,000 batches, complete) differs in:",
            "Batches: 777 here, 4,000 in this set", "Simulations per batch: 32 here, 1,024 in this set",
            f"Rotation: {words[rotate]} here, {words[not rotate]} in this set",
            "Simulation set for retrain_prior (30 batches, complete) differs in:",
            f"Bounds file: lower limit of {first}: {_exact(mine[0])} here, 0.75 in this set",
            f"Bounds file: upper limit of {last}: {_exact(mine[1])} here, 99.5 in this set",
            f"Simulation set for {other_label} (12 batches, partial) differs in:",
            f"Prior: retrain_prior here, {other_label} in this set"), question.items
        assert other_label.startswith("Unnamed prior, ")
        assert question.beneath == "To continue one of them, cancel and set these back."
        assert stand_in.calls == [], "Cancel started the training"
        choose(monkeypatch, words="Start training on a new simulation set")
        page._main_clicked()
        _the_call(stand_in, 1)
        _args, kwargs = stand_in.calls[-1]
        assert (kwargs["new_run"], kwargs["num_runs"], kwargs["run_size_cap"], kwargs["name"]) == (True, 777, 32,
                                                                                                    "retrain_2")
        stand_in.release(None)
        _settle_the_run()
        forget_asked()
        monkeypatch.setattr(training_checkpoint, "peek",
                            lambda path: (_ for _ in ()).throw(OSError("the disk said no")))
        page.names.name.setText("retrain_3")
        page.read_facts()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert "Attention. The saved simulation sets could not be read: OSError: the disk said no." in said, said
        assert page.foot.main_button.isEnabled()
        page._main_clicked()
        assert all_asked() == [], "a reading that could not read the sets asked a question"
        _the_call(stand_in, 2)
        assert stand_in.calls[-1][1]["new_run"] is False
        stand_in.release(None)
        _settle_the_run()


def _exact(value) -> str:
    """A fraction as a difference says it: in the shortest form that is exactly it, a whole one without its ".0"."""
    said = repr(float(value))
    return said[:-2] if said.endswith(".0") else said


def _the_call(stand_in, count) -> None:
    """Turn the event loop until the stand-in stage has been called ``count`` times in all."""
    from tests._runs import turn_until
    turn_until(qt_app(), lambda: len(stand_in.calls) == count, 10.0)


def _settle_the_run() -> None:
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    turn_until(qt_app(), lambda: not controller().is_live(), 10.0)


def test_a_set_of_another_model_is_not_listed_and_a_rounds_set_says_so(tmp_path, monkeypatch):
    """A saved set made for another model is that model's, and the question does not list it; with no other set saved
    beside it, nothing is asked and the training starts without the consent. A set a narrowing round made is listed like
    any other, and says that a training never continues it."""
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page)
        _a_set_beside(page, 900, model="HOPF")
        page.read_facts()
        page._main_clicked()
        _the_call(stand_in, 1)
        assert all_asked() == [] and stand_in.calls[-1][1]["new_run"] is False
        stand_in.release(None)
        _settle_the_run()
        page.names.name.setText("retrain_3")
        _a_set_beside(page, 40, truncation={"level": 0.95, "dims": [0, 1]})
        page.read_facts()
        page._main_clicked()
        assert asked().items == ("Simulation set for retrain_prior (40 batches, complete) differs in:",
                                 "Made by a narrowing round: a training never continues it."), asked().items
        assert len(stand_in.calls) == 1, "Cancel started the training"


def test_with_never_the_question_lists_the_set_it_will_not_continue(tmp_path, monkeypatch):
    """With "Never" the simulations are not saved and nothing is continued, so the sets that exist are read once more at
    the cadence the saves are usually made at, with the batches and the simulations per batch typed: a set these very
    settings would continue heads the list, as one no setting differs from, and the question then says none will be
    continued and its answer that the simulations are not saved; with no such set, the others of the model are listed.
    The answer starts the training with no saves and the consent."""
    from core import config, orchestrator
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page)
        own = _a_set_beside(page, 300, complete=False)
        rows = page.form.rows
        rows["checkpoint_every"].widget.field.setText("25")
        rows["checkpoint_every"].widget.ticked_box.setChecked(True)
        read, kept = [], orchestrator.training_reading
        monkeypatch.setattr(orchestrator, "training_reading", lambda *args, **kwargs: (
            read.append({key: kwargs[key] for key in ("num_runs", "run_size_cap", "checkpoint_every")}),
            kept(*args, **kwargs))[1])
        page.read_facts()
        assert read == [{"num_runs": 777, "run_size_cap": 32, "checkpoint_every": 0},
                        {"num_runs": 777, "run_size_cap": 32, "checkpoint_every": config.TRAINING_CHECKPOINT_EVERY}]
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert ("Attention. The simulations of this training are not saved, so no saved simulation set is continued."
                in said), said
        page._main_clicked()
        question = asked()
        assert question.sentence == ("This training will simulate 777 batches from zero. Saved simulation sets for "
                                     "this "
                                     "model exist, and none will be continued.")
        assert question.items == (
            "Simulation set for retrain_prior (300 batches, partial):",
            "No setting differs. It is not continued because the simulations of this training are not saved.")
        assert question.answers == ("Start training without saving the simulations", "Cancel")
        assert question.default == "Cancel" and stand_in.calls == []
        import shutil
        from core.artifacts import default_store
        shutil.rmtree(default_store().kind_dir("simulation") / own)
        rows["checkpoint_every"].widget.ticked_box.setChecked(False)
        _a_set_beside(page, 4000, n_runs=4000)
        rows["checkpoint_every"].widget.ticked_box.setChecked(True)
        page.read_facts()
        choose(monkeypatch, words="Start training without saving the simulations")
        page._main_clicked()
        assert asked().items == ("Simulation set for retrain_prior (4,000 batches, complete) differs in:",
                                 "Batches: 777 here, 4,000 in this set"), asked().items
        _the_call(stand_in, 1)
        _args, kwargs = stand_in.calls[-1]
        assert (kwargs["checkpoint_every"], kwargs["new_run"], kwargs["resume"]) == (0, True, "auto")
        stand_in.release(None)
        _settle_the_run()


#: The stage's own sentence when it refuses a new set beside a saved one a setting away, as it words it.
NEAR_MISS = ("This run would start a NEW simulation cache from zero, but a committed cache ONE setting away exists. If "
             "you meant to continue that cache, set n_runs back to 4000. To start a new cache anyway, pass "
             "new_run=True.")


def test_a_new_run_refusal_asks_the_question_and_shows_no_engine_sentence(tmp_path, monkeypatch):
    """A set saved one setting away between the press and the stage's entry: the stage refuses the new set, and the
    page, in place of a box, reads again and asks the question before simulating from zero, which lists that set; its
    answer starts the training again, its call the first's with the consent. "Cancel" starts nothing. Either way, the
    stage's sentence is in no box, no question and no fact."""
    from core.refusals import Refusal
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page)
        page.read_facts()
        for answer in ("Start training on a new simulation set", "Cancel"):
            forget_asked()
            choose(monkeypatch, words=answer)
            before = len(stand_in.calls)
            page._main_clicked()
            _the_call(stand_in, before + 1)
            assert all_asked() == []
            first_args, first_kwargs = stand_in.calls[-1]
            assert first_kwargs["new_run"] is False
            name = _a_set_beside(page, 4000, n_runs=4000)
            stand_in.fail(Refusal(NEAR_MISS, field="new_run"))
            if answer == "Cancel":
                _settle_the_run()
                assert len(stand_in.calls) == before + 1, "Cancel started the training"
            else:
                from core.gui.run.controller import controller
                from tests._runs import turn_until
                turn_until(qt_app(), lambda: len(stand_in.calls) == before + 2 and controller().is_live(), 10.0)
                assert stand_in.calls[-1] == (first_args, {**first_kwargs, "new_run": True})
                stand_in.release(None)
                _settle_the_run()
            question, = all_asked()
            assert question.title == "Start a new simulation set?" and question.items == (
                "Simulation set for retrain_prior (4,000 batches, complete) differs in:",
                "Batches: 777 here, 4,000 in this set"), (answer, question.items)
            told = [question.sentence, question.beneath, *question.items, page.in_use.said.sentence.plain_text(),
                    *(row.sentence.plain_text() for row in page.foot.facts.rows())]
            assert not [words for words in told if "new_run" in words or "NEW simulation cache" in words], told
            import shutil
            from core.artifacts import default_store
            shutil.rmtree(default_store().kind_dir("simulation") / name)
            page.names.name.setText("retrain_3")
            page.read_facts()


@pytest.mark.parametrize("which", ["another model's set", "a reading that cannot read the sets"])
def test_a_new_run_refusal_with_no_set_listed_still_asks_and_shows_no_engine_sentence(which, tmp_path, monkeypatch):
    """The stage refuses a new set while the page's own reading lists no set beside it, as when the only set one setting
    away is another model's, or the saved sets cannot be read: the page still asks the question before simulating from
    zero, its list empty, and its answer starts the training again with the consent; the stage's sentence is in no box,
    no question and no fact."""
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from core.SBI import training_checkpoint
    from tests._runs import turn_until
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _a_budget(page)
        page.read_facts()
        page._main_clicked()
        _the_call(stand_in, 1)
        first_args, first_kwargs = stand_in.calls[-1]
        if which == "another model's set":
            _a_set_beside(page, 4000, n_runs=4000, model="HOPF")
        else:
            monkeypatch.setattr(training_checkpoint, "peek",
                                lambda path: (_ for _ in ()).throw(OSError("the disk said no")))
        choose(monkeypatch, words="Start training on a new simulation set")
        stand_in.fail(Refusal(NEAR_MISS, field="new_run"))
        turn_until(qt_app(), lambda: len(stand_in.calls) == 2 and controller().is_live(), 10.0)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "new_run": True})
        stand_in.release(None)
        _settle_the_run()
        question, = all_asked()
        assert (question.title, question.items, question.answers) == (
            "Start a new simulation set?", (), ("Start training on a new simulation set", "Cancel")), question
        told = [question.sentence, question.beneath, *question.items, page.in_use.said.sentence.plain_text(),
                *(row.sentence.plain_text() for row in page.foot.facts.rows())]
        assert not [words for words in told if "new_run" in words or "NEW simulation cache" in words], told


def test_the_consent_to_a_new_set_travels_with_every_repeat_of_the_call(tmp_path, monkeypatch):
    """A training started with the consent to start a new simulation set beside a saved one hands that consent on with
    its own call: the window's try after the graphics card ran out of memory before a batch was saved repeats it, which
    the stage would otherwise refuse beside that set, as the stand-in stage refuses it here; "Continue from batch N" and
    "Resume the training" repeat it too, where it changes nothing the stage does."""
    import torch
    from core import orchestrator
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, to_the_simulation, turn_until
    app = qt_app()
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True)
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        held, view = controller(), page.run_view
        text = RunText(held)
        _a_budget(page)
        _a_set_beside(page, 4000, n_runs=4000)
        refused = []

        def stage(*args, **kwargs):
            if not kwargs.get("new_run") and kwargs.get("resume") != "require" and kwargs.get("resume_fit") is None:
                refused.append(kwargs)
                raise Refusal(NEAR_MISS, field="new_run")
            return stand_in(*args, **kwargs)

        monkeypatch.setattr(orchestrator, "build_posterior", stage)
        choose(monkeypatch, words="Start training on a new simulation set")
        page.read_facts()
        page._main_clicked()
        _the_call(stand_in, 1)
        first_args, first_kwargs = stand_in.calls[-1]
        assert first_kwargs["new_run"] is True
        lack = RuntimeError("the simulation of batch 1 failed")
        lack.__cause__ = RuntimeError("CUDA error: out of memory")
        stand_in.fail(lack)
        turn_until(app, lambda: held.status.state == "retrying", 10.0)
        held._clock.advance(60)
        held.tick()
        turn_until(app, lambda: len(stand_in.calls) == 2 and held.status.state == "running", 10.0)
        assert stand_in.calls[-1] == (first_args, first_kwargs)
        torch.save({"batches_done": 236, "complete": False}, folder / "state.pt")
        to_the_simulation(app, stand_in, text, held, folder, 236)
        stand_in.fail(OSError(28, "No space left on device"))
        turn_until(app, lambda: not held.is_live(), 10.0)
        view.continue_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 3)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "require"})
        kept = build_fit_record(default_store(), passes=3)
        stand_in.kept_fit = note(kept, passes=3)
        to_the_fit(app, stand_in, text, held._clock, folder, ((1, 0), (2, 1), (3, 2)), again=False)
        choose(monkeypatch, words="Stop training")
        view.stop_button.click()
        turn_until(app, lambda: not held.is_live(), 10.0)
        view.resume_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 4)
        assert stand_in.calls[-1] == (first_args, {**first_kwargs, "resume": "require", "resume_fit": kept})
        assert refused == [], "a repeat of the call lost its consent"
        stand_in.release(None)
        turn_until(app, lambda: not held.is_live(), 10.0)


def test_a_blank_rotation_setting_does_not_stop_a_training_that_will_not_read_it(tmp_path, monkeypatch):
    """The rotation's three are read only while the rotation is computed afresh. With the rotation off, or a saved set of
    these settings continued, whose own rotation is reused, they are greyed with the reason under them, are not judged,
    and are not passed: a blank one stops nothing. With the rotation computed afresh a blank one is judged and greys the
    start. A box of the budget left blank sets the last reading aside, at once: what it found of the saved set is not
    said, and the rotation's three are read and judged again."""
    from core.gui.run.controller import RunController
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        seen = []
        monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.append(kwargs))
        rows = page.form.rows
        rows["fisher_dz"].widget.setText("")
        page.read_facts()
        assert rows["fisher_dz"].line.words == "Rotation: step is blank." and not page.foot.main_button.isEnabled()
        assert page.rotation_line.isHidden() and rows["fisher_dz"].widget.isEnabled()
        for why, words in (("off", "Not used: rotation is off."),
                           ("a saved set", "Not used: this training reuses the saved simulation set's rotation.")):
            if why == "off":
                screen.session.cfg.reparam_rotate = False
            else:
                screen.session.cfg.reparam_rotate = True
                rows["num_runs"].widget.setText("6000")
                _a_saved_set(page, 12)
            page.read_facts()
            assert (page.rotation_line.isHidden(), page.rotation_line.plain_text()) == (False, words), why
            assert not any(rows[key].widget.isEnabled() for key in ("fisher_m", "fisher_dz", "fisher_points")), why
            assert rows["fisher_dz"].line.words == "" and page.foot.main_button.isEnabled(), why
            page._main_clicked()
            assert not {"fisher_m", "fisher_dz", "fisher_points"} & set(seen[-1]), (why, seen[-1])
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert any("12 of 6,000 batches are done" in one for one in said), said
        rows["checkpoint_every"].widget.field.setText("")
        page.redraw()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert not any("batches are done" in one for one in said), "a reading made at another budget was said"
        assert page.rotation_line.isHidden() and all(rows[key].widget.isEnabled()
                                                     for key in ("fisher_m", "fisher_dz", "fisher_points")), (
            "the rotation's three followed a reading the facts set aside")
        assert rows["fisher_dz"].line.words == "Rotation: step is blank.", "the rotation's three were not judged"


def test_never_fixes_continue_or_start_new_with_its_reason(tmp_path, monkeypatch):
    """"Never" for the saves, or a 0 typed for them: "Continue or start new" goes back to its first choice and is greyed,
    with the reason under it; the start passes that first choice, whatever was chosen before. Ticked off, or another
    number typed, the choice can be made again."""
    from core.gui.run.controller import RunController
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        seen = []
        monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.append(kwargs))
        never = page.form.rows["checkpoint_every"].widget
        _pick(page.resume_choice, "never")
        assert page.never_line.isHidden()
        never.ticked_box.setChecked(True)
        assert (page.resume_choice.currentData(), page.resume_choice.isEnabled(), page.never_line.isHidden()) == (
            "auto", False, False)
        assert page.never_line.plain_text() == (
            "No simulation set is kept or continued. A stop or a failure during simulation loses every simulation; a "
            "stop or a failure during the fit loses the fit, which cannot be tried again without simulating again; "
            "and the check 'Use of each part of the input' cannot be run on the posterior.")
        page.read_facts()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert ("Attention. The simulations of this training are not saved, so no saved simulation set is continued."
                in said and "Attention. A training whose simulation set is not kept cannot be resumed: a stop or a "
                "failure during the fit loses it." in said), said
        page._main_clicked()
        assert (seen[-1]["resume"], seen[-1]["checkpoint_every"]) == ("auto", 0)
        never.ticked_box.setChecked(False)
        assert page.resume_choice.isEnabled() and page.never_line.isHidden()
        _pick(page.resume_choice, "require")
        page.read_facts()
        assert page.foot.reason.text() == "Cannot start: there is nothing to continue."
        never.field.setText("0")                    # a typed 0 saves nothing either, as the stage reads it
        assert (page.resume_choice.currentData(), page.resume_choice.isEnabled(), page.never_line.isHidden()) == (
            "auto", False, False)
        never.field.setText("50")
        assert page.resume_choice.isEnabled() and page.never_line.isHidden()


def test_the_expected_epochs_reach_the_run_and_never_the_stage(tmp_path, monkeypatch):
    """"Expected epochs" is for the time alone: typed, it reaches the run's page through its spec and the fit's time in
    "Before you start", and never the stage; left blank, the last finished training of this kind is read, and with none
    the line under the field says what the run's page will do, once the page has looked: not before its reading has
    asked, with no prior held or a box of the budget out of its rule. A finished training of this kind in Records gives
    its epochs to the fit's time while the box is blank; typed epochs win, and cleared, the training's come back."""
    from datetime import datetime, timedelta, timezone
    from core.artifacts import default_store
    from core.gui.run.controller import RunController, controller
    from tests._fixtures import _posterior_artifact, write_history
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        seen = []
        monkeypatch.setattr(RunController, "dispatch", lambda held, fn, *args, **kwargs: seen.append(kwargs))
        assert not page.no_earlier_line.isHidden(), "no earlier training, and the line does not say so"
        page.form.rows["num_runs"].widget.setText("0")            # a box of the budget out of its rule
        page.read_facts()
        assert page.no_earlier_line.isHidden(), "the page said there is none without looking"
        page.form.rows["num_runs"].widget.setText("5000")
        prior = screen.session.inf_prior
        screen.session.inf_prior = None                           # no prior held: no training of any kind to look for
        page.read_facts()
        assert page.no_earlier_line.isHidden(), "the page said there is none without looking"
        screen.session.inf_prior = prior
        page.read_facts()
        assert not page.no_earlier_line.isHidden()
        page.form.rows["expected_epochs"].widget.setText("150")
        page.read_facts()
        assert page.no_earlier_line.isHidden()
        page._main_clicked()
        assert "expected_epochs" not in seen[-1] and seen[-1]["spec"].expected_epochs == 150
        page.form.rows["expected_epochs"].widget.setText("")
        assert not page.no_earlier_line.isHidden(), "the epochs once typed stood for the last training's"
        page.read_facts()
        page._main_clicked()
        assert seen[-1]["spec"].expected_epochs is None
        width = page._reading.width
        started = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        write_history(controller().folder(), id="20261008T120000-training", word="Training", name="retrain_1",
                      outcome="finished", started=started, ended=started + timedelta(hours=3),
                      simulation_set={"per_batch": width, "seconds_per_batch": 9.7},
                      fit={"network": [128, 8], "simulations": 5000 * width, "seconds_per_epoch": 60.0})
        _posterior_artifact(default_store(), screen.session.cfg, name="retrain_1",
                            over={("training", "epochs_trained"): 130})
        page.read_facts()

        def fit_time():
            return next(row.sentence.plain_text() for row in page.foot.facts.rows()
                        if "Fit: " in row.sentence.plain_text())

        assert page.no_earlier_line.isHidden() and " for 130 epochs: " in fit_time(), fit_time()
        page.form.rows["expected_epochs"].widget.setText("150")
        page.redraw()
        assert " for 150 epochs: " in fit_time(), fit_time()
        page.form.rows["expected_epochs"].widget.setText("")
        page.redraw()
        assert page.no_earlier_line.isHidden() and " for 130 epochs: " in fit_time(), fit_time()


def test_the_name_and_note_are_emptied_after_a_record_and_kept_after_a_stop(tmp_path, monkeypatch):
    """A training that returned its posterior empties the name and the note, so a second press starts nothing until a new
    name is typed; one stopped before it returned one keeps them as typed. The record a training made is let go at its
    run's end: a later run that ends any other way, a training stopped or another page's run, leaves the name and the
    note typed since and says nothing of a name taken."""
    from core.gui.run.controller import controller
    from tests._fixtures import build_run_host
    from tests._runs import running, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        choose(monkeypatch, words="Stop training")
        page._main_clicked()
        assert stand_in.started.wait(5)
        page.run_view.stop_button.click()
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_2", "a second try")
        page.run_view.back_button.click()
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="posterior", id="p1", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert (page.names.name.text(), page.names.note.text()) == ("", "")
        assert not page.foot.main_button.isEnabled() and page.foot.reason.text() == "Cannot start yet: a name is needed."
        page.run_view.back_button.click()
        page.names.name.setText("retrain_3")
        page.names.note.setText("a third try")
        page._main_clicked()
        assert stand_in.started.wait(5)
        page.run_view.stop_button.click()
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_3", "a third try")
        assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()
        with running(build_run_host()):
            pass
        assert (page.names.name.text(), page.names.note.text()) == ("retrain_3", "a third try")
        assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()


def test_a_record_saved_under_another_name_says_so(tmp_path, monkeypatch):
    """A posterior that comes back under no name, its name taken while the run was going, says so under what is in use,
    in the run's own sentence; one that came back under its name says nothing."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="posterior", id="p1", name="", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        said = page.in_use.said
        assert not said.isHidden() and said.sentence.plain_text() == (
            "Attention. It was saved without its name: 'retrain_2' was taken while the run was going.")
        page.names.name.setText("retrain_3")
        page.read_facts()
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release(SimpleNamespace(kind="posterior", id="p2", name="retrain_3", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert said.isHidden(), said.sentence.plain_text()


def test_a_record_is_judged_by_the_name_its_run_asked_for_and_a_name_typed_since_stands(tmp_path, monkeypatch):
    """A training stopped in its fit and resumed from its run's page repeats its own call, its name with it: the record
    that comes back under that name says nothing of a name taken, though another name and note were typed meanwhile, and
    those stand; only fields that still hold what the call was handed are emptied."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, turn_until
    app = qt_app()
    folder = tmp_path / "simulations" / "_unnamed__20261003T030211"
    folder.mkdir(parents=True)
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        held, view = controller(), page.run_view
        text = RunText(held)
        page._main_clicked()
        assert stand_in.started.wait(5)
        kept = build_fit_record(default_store(), passes=2)
        stand_in.kept_fit = note(kept, passes=2)
        to_the_fit(app, stand_in, text, held._clock, folder, ((1, 0), (2, 1)))
        choose(monkeypatch, words="Stop training")
        view.stop_button.click()
        turn_until(app, lambda: not held.is_live(), 10.0)
        page.names.name.setText("another_name")
        page.names.note.setText("typed since")
        view.resume_button.click()
        turn_until(app, lambda: len(stand_in.calls) == 2 and held.is_live(), 10.0)
        assert (stand_in.calls[-1][1]["name"], stand_in.calls[-1][1]["note"]) == ("retrain_2", "a second try")
        stand_in.release(SimpleNamespace(kind="posterior", id="p1", name="retrain_2", manifest=None))
        turn_until(app, lambda: not held.is_live(), 10.0)
        assert page.in_use.said.isHidden(), page.in_use.said.sentence.plain_text()
        assert (page.names.name.text(), page.names.note.text()) == ("another_name", "typed since")


def test_each_of_the_name_and_the_note_is_emptied_on_its_own_after_a_resume(tmp_path, monkeypatch):
    """A training stopped in its fit and resumed from its run's page with nothing typed meanwhile returns its record
    under the name its call asked for, and both fields, which still hold what the call was handed, are emptied. Stopped
    again and resumed with the note alone typed anew, its record empties the name, which still holds what the call was
    handed, and leaves the note typed since."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import build_fit_record
    from tests._runs import RunText, note, to_the_fit, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        held, view = controller(), page.run_view

        def stopped_in_the_fit(folder, started):
            text = RunText(held)
            page._main_clicked()
            assert stand_in.started.wait(5)
            kept = build_fit_record(default_store(), passes=2)
            stand_in.kept_fit = note(kept, passes=2)
            to_the_fit(app, stand_in, text, held._clock, folder, ((1, 0), (2, 1)))
            choose(monkeypatch, words="Stop training")
            view.stop_button.click()
            turn_until(app, lambda: not held.is_live(), 10.0)
            assert len(stand_in.calls) == started

        def resumed(made):
            view.resume_button.click()
            turn_until(app, lambda: held.is_live(), 10.0)
            stand_in.release(SimpleNamespace(kind="posterior", id=made, name=stand_in.calls[-1][1]["name"],
                                             manifest=None))
            turn_until(app, lambda: not held.is_live(), 10.0)

        for at in (1, 2):
            (tmp_path / "simulations" / f"_unnamed__2026100{at}T030211").mkdir(parents=True)
        stopped_in_the_fit(tmp_path / "simulations" / "_unnamed__20261001T030211", 1)
        resumed("p1")
        assert stand_in.calls[-1][1]["name"] == "retrain_2"
        assert (page.names.name.text(), page.names.note.text()) == ("", "")
        page.run_view.back_button.click()
        page.names.name.setText("retrain_3")
        page.names.note.setText("a third try")
        page.read_facts()
        stopped_in_the_fit(tmp_path / "simulations" / "_unnamed__20261002T030211", 3)
        page.names.note.setText("typed since")
        resumed("p2")
        assert (stand_in.calls[-1][1]["name"], stand_in.calls[-1][1]["note"]) == ("retrain_3", "a third try")
        assert (page.names.name.text(), page.names.note.text()) == ("", "typed since")


def test_a_taken_name_greys_the_start_as_it_is_typed(tmp_path, monkeypatch):
    """A name another posterior holds is said under "Name" once the typing stops, in the store's own words, and greys
    "Start training" with its words beside it; a free name says so and the button comes back."""
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _posterior_artifact
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        _posterior_artifact(default_store(), _nad_cfg(), name="taken")
        page.names.name.setText("taken")
        page.names.check_now()
        row = page.form.rows["name"]
        assert row.line.kind == "err" and "taken" in row.line.words, row.line.words
        assert not page.foot.main_button.isEnabled() and page.foot.reason.text() == (
            "Cannot start yet: the name cannot be used.")
        page.names.name.setText("free_one")
        page.names.check_now()
        assert row.line.words == "Fine. This name is free." and page.foot.main_button.isEnabled()


def test_the_number_of_simulations_is_said_in_words(tmp_path, monkeypatch):
    """"Before you start" says how many simulations a training makes: the batches typed, times the simulations of each
    as the reading at this budget gives them, which a cap typed under the computer's own batch lowers; and what each
    simulates. The boxes open at config.py's values, never at a session's."""
    from core import config
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        rows = page.form.rows
        assert (rows["num_runs"].value(), rows["run_size_cap"].value()) == (config.TRAINING_NUM_RUNS,
                                                                           config.TRAINING_RUN_SIZE)
        width = page._reading.width
        assert width == screen.session.cfg.hw.batch_size

        def said():
            return [row.sentence.plain_text() for row in page.foot.facts.rows()][1]

        assert said() == (f"Information. {config.TRAINING_NUM_RUNS:,} batches of {width:,} make "
                          f"{config.TRAINING_NUM_RUNS * width:,} simulations. Each simulates one spontaneous recording "
                          "and one driven recording.")
        rows["run_size_cap"].widget.ticked_box.setChecked(False)
        rows["run_size_cap"].widget.field.setText(str(width // 4))
        rows["num_runs"].widget.setText(str(config.TRAINING_NUM_RUNS * 4))
        page.read_facts()
        assert said() == (f"Information. {config.TRAINING_NUM_RUNS * 4:,} batches of {width // 4:,} make "
                          f"{config.TRAINING_NUM_RUNS * width:,} simulations. Each simulates one spontaneous recording "
                          "and one driven recording.")


def test_the_sentences_beside_the_settings_are_the_pages_own(tmp_path, monkeypatch):
    """What stands beside the settings, word for word: under the network, that width and depth open at their usual values
    at each launch; under the epoch limit, its two traps; under the expected epochs, that they never limit the fit, and,
    with no earlier training of this kind, what the run's page will say; under the choice of continuing, nothing until
    the simulations are not saved; under the rotation's three, nothing while they are read."""
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        page.form.fold.set_open(True)

        def under(key):
            return [part.plain_text() for part in page.form.rows[key].under if not part.isHidden()]

        assert under("num_transforms") == ["Width and depth open at 128 and 8 each time PRISM starts. A plan can set "
                                           "them."]
        assert under("max_num_epochs") == ["A limit of N lets the fit run N + 1 epochs.",
                                           "A fit that reaches the limit keeps the network of its last epoch, which "
                                           "may not be its best."]
        assert under("expected_epochs") == [
            "Used only to say when the fit should end. It never limits the fit. Left blank, PRISM takes the epochs of "
            "the last finished training of this kind in Records.",
            "No earlier training of this kind: the run's page will say only when the fit cannot end before."]
        assert under("resume") == [] and under("fisher_points") == []


def test_a_history_no_reading_foresaw_says_no_time_and_raises_nothing(tmp_path, monkeypatch):
    """A run history written by hand may hold a pace that is no number: the page's reading says no time is known and
    raises nothing, and the press still starts the training."""
    from datetime import datetime, timedelta, timezone
    from core.gui.run.controller import controller
    from tests._fixtures import write_history
    from tests._runs import turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        started = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        width = page._reading.width
        write_history(controller().folder(), id="20261008T120000-training", word="Training", name="by_hand",
                      outcome="finished", started=started, ended=started + timedelta(hours=1),
                      simulation_set={"per_batch": width, "seconds_per_batch": "fast"},
                      fit={"network": [128, 8], "simulations": page.form.rows["num_runs"].value() * width,
                           "seconds_per_epoch": "slow"})
        page.read_facts()
        said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
        assert ("Information. No time estimate: no training of this size has run on this computer, and the timing of "
                "a short fit has not been run.") in said, said
        page._main_clicked()
        assert stand_in.started.wait(5)
        stand_in.release(None)
        turn_until(app, lambda: not controller().is_live(), 10.0)


# ── the checks the Training page reads ──
@contextlib.contextmanager
def _third_step_of_the_frequency_response(tmp_path, monkeypatch):
    """The third step as ``_third_step`` builds it with no posterior held, its configuration and its prior of the
    frequency response."""
    import tests._fixtures as fixtures
    real = fixtures._nad_cfg
    monkeypatch.setattr(fixtures, "_nad_cfg", lambda **over: real(**{"chi_mode": True, **over}))
    try:
        with _third_step(tmp_path, monkeypatch, held=False) as built:
            yield built
    finally:
        monkeypatch.setattr(fixtures, "_nad_cfg", real)


def test_a_bounds_file_in_a_subfolder_of_its_models_folder_is_selected_by_a_chain_of_loads(tmp_path, monkeypatch):
    """A chain of loads names its bounds file by its path in its model's folder: with forward slashes, as a kept
    training's record gives it to a Resume, or with the computer's own separator, as a run's history gives it to
    "Continue from batch N" and "Fit again" and the key remembered from last time gives it to Home's "Continue with
    these". Either way a file in a subfolder of that folder is selected in step 2's list, whose entries are spelled
    with the computer's own separator, as a file at the top of the folder is; Home's own press selects it too."""
    import os
    import shutil
    from core.config import BOUNDS_PATH
    from core.gui import remembered
    from core.gui.panels.inference import prior_tab
    folder = tmp_path / "Bounds" / "nadrowski"
    (folder / "sub").mkdir(parents=True)
    for path in (folder / "master.txt", folder / "sub" / "x.txt"):
        shutil.copyfile(BOUNDS_PATH / "nadrowski" / "master.txt", path)
    monkeypatch.setattr(prior_tab, "BOUNDS_PATH", tmp_path / "Bounds")
    screen = build_inference()
    beneath, selected = os.path.join("sub", "x.txt"), []
    for named in ("master.txt", "sub/x.txt", "master.txt", beneath, "master.txt"):
        screen.load_chain("NADROWSKI", {"chi_mode": False}, named, None)
        selected.append(screen.prior_panel.bounds_picker.key())
    forget_asked()
    screen.continue_with(remembered.Picks(model="NADROWSKI", chi_mode=False, bounds=beneath))
    selected.append(screen.prior_panel.bounds_picker.key())
    assert (selected, all_asked()) == (["master.txt", beneath, "master.txt", beneath, "master.txt", beneath], []), (
        selected)


def _an_audit(screen, *, prior=None, created=None, **given) -> str:
    """An audit of the prior in use (or of ``prior``, an id), as the probe check writes one on the configuration in use,
    reading as ``given`` makes it; its id."""
    from core.artifacts import default_store
    from core.diagnostics.probes import configured_chi_settings
    from tests._fixtures import check_results, write_diagnostic
    cfg = screen.session.cfg
    held = check_results("audit", **given)
    settings = {**held["settings"], "configured": {**configured_chi_settings(cfg),
                                                   "probe_count": {"drawn_per_batch_from": [2, 12]}}}
    return write_diagnostic(default_store(), cfg, diagnostic="probes", variant="mask",
                            parents={"prior": prior or screen.session.inf_prior.id}, settings=settings,
                            results=held["results"], created=created)


def _a_band(screen, *, created=None, **given) -> str:
    """A frequency-band record made on the configuration in use, reading as ``given`` makes it; its id."""
    from core.artifacts import default_store
    from core.diagnostics.probes import configured_chi_settings
    from tests._fixtures import check_results, write_diagnostic
    cfg = screen.session.cfg
    held = check_results("band", **given)
    return write_diagnostic(default_store(), cfg, diagnostic="probes", variant="band",
                            settings={**held["settings"], "configured": configured_chi_settings(cfg)},
                            results=held["results"], created=created)


def _checks_line(page) -> "tuple[int, str] | None":
    """Where "Before you start" says the two checks made before a training, and what it says; None when it does not."""
    for at, row in enumerate(page.foot.facts.rows()):
        said = row.sentence.plain_text()
        if "Frequency band:" in said or "pre-run checks" in said:
            return at, said
    return None


def test_the_training_page_says_the_pre_run_checks_as_it_reads_them(tmp_path, monkeypatch):
    """"Before you start" on the Training page says what the frequency band and the audit of the prior in use read, as
    the page's own reading finds them, after the memory limit and before the other steps' changes: "Attention." with
    neither run; each as its record is written and the store changes, with its date; "Fine." with both passed; an audit
    of another prior moves nothing. It never greys "Start training". With a configuration that is not of the frequency
    response, "Information.". With no prior in use it says nothing of them."""
    from core.gui.design import date_text
    from tests._runs import at
    qt_app()
    with _third_step_of_the_frequency_response(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        place, said = _checks_line(page)
        memory = next(n for n, row in enumerate(page.foot.facts.rows())
                      if "memory" in row.sentence.plain_text().lower())
        assert place > memory, (place, memory)
        assert said == "Attention. Frequency band: Not run yet. Unusable driven recordings: audit: Not run yet."
        assert page.foot.main_button.isEnabled()
        _an_audit(screen, prior="20201010T101010", masked_fraction=0.5)
        window._store_changed()
        assert _checks_line(page)[1] == said, "an audit of another prior was counted"
        _an_audit(screen, created=at(3, 14, 40))
        window._store_changed()
        assert _checks_line(page)[1] == "Attention. Frequency band: Not run yet."
        _a_band(screen, created=at(3, 14, 2))
        window._store_changed()
        assert _checks_line(page)[1] == (
            f"Fine. Frequency band: Passed, {date_text(at(3, 14, 2))}. Unusable driven recordings: audit: Passed, "
            f"{date_text(at(3, 14, 40))}.")
        _a_band(screen, created=at(4, 9, 0), band_holds=False)
        window._store_changed()
        assert _checks_line(page)[1] == (
            f"Attention. Frequency band: Failed, {date_text(at(4, 9, 0))}. Starting will ask first.")
        assert page.foot.main_button.isEnabled(), "a failed check greyed the start"
        prior, screen.session.inf_prior = screen.session.inf_prior, None
        page.read_facts()
        assert _checks_line(page) is None, "the pre-run checks were said with no prior in use"
        screen.session.inf_prior = prior
        import tests._fixtures as fixtures
        screen.session.cfg = fixtures._nad_cfg(chi_f0=0.2)
        assert [row.state for row in screen.check_rows()[:2]] == ["Not run yet", "Not run yet"], (
            "the records made at another drive amplitude counted for a configuration put in use since")
    with _third_step(tmp_path / "plain", monkeypatch, held=False) as (window, screen, page, stand_in):
        assert _checks_line(page)[1] == ("Information. The two pre-run checks do not apply: they need the frequency "
                                         "response.")


def test_a_failed_pre_run_check_asks_first_with_cancel_preselected(tmp_path, monkeypatch):
    """After the audit of the prior in use read "Failed", "Start training" asks one question first, "Cancel"
    preselected: its title and its other answer from the button's words, its text naming the check that failed with its
    date and where to read why. "Cancel" starts nothing; "Start training anyway" goes on to the question before
    simulating from zero, and asks nothing more with no saved set beside; when the stage then refuses a new set beside a
    saved one, its own question is asked again and the failed check's is not. Once the check passes, nothing is asked.
    After it fails again, a press on "Continue from batch 236" asks in those words, and a kept training's Resume asks
    nothing."""
    from core.gui.design import date_text
    from core.refusals import Refusal
    from tests._dialogs import answer_in_turn
    from tests._runs import at
    qt_app()
    with _third_step_of_the_frequency_response(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        _a_budget(page)
        _an_audit(screen, created=at(3, 14, 40), masked_fraction=0.5)
        window._store_changed()
        title, text = ("Start training although a check failed?",
                       f"Unusable driven recordings: audit: Failed, {date_text(at(3, 14, 40))}. Step 4, Checks, says "
                       "why.")
        forget_asked()
        choose(monkeypatch, default=True)
        page._main_clicked()
        question = asked()
        assert (question.title, question.sentence, question.answers, question.default) == (
            title, text, ("Start training anyway", "Cancel"), "Cancel"), question
        assert stand_in.calls == [] and len(all_asked()) == 1
        name = _a_set_beside(page, 40, n_runs=40)
        forget_asked()
        answer_in_turn(monkeypatch, ("Start training anyway", None), ("Cancel", None))
        page._main_clicked()
        assert [one.title for one in all_asked()] == [title, "Start a new simulation set?"]
        assert stand_in.calls == []
        import shutil
        from core.artifacts import default_store
        shutil.rmtree(default_store().kind_dir("simulation") / name)
        page.read_facts()
        forget_asked()
        answer_in_turn(monkeypatch, ("Start training anyway", None), ("Cancel", None))
        page._main_clicked()
        _the_call(stand_in, 1)
        assert [one.title for one in all_asked()] == [title], "a question was asked after the failed check's"
        assert stand_in.calls[-1][1]["new_run"] is False
        _a_set_beside(page, 4000, n_runs=4000)
        stand_in.fail(Refusal(NEAR_MISS, field="new_run"))
        _settle_the_run()
        assert [one.title for one in all_asked()] == [title, "Start a new simulation set?"], (
            "the failed check's question was asked again at the stage's refusal of a new set")
        assert len(stand_in.calls) == 1
        _an_audit(screen, created=at(4, 9, 0))
        window._store_changed()
        forget_asked()
        choose(monkeypatch, words="Cancel")
        page._main_clicked()
        assert title not in [one.title for one in all_asked()], "a check that passed asked first"
        _an_audit(screen, created=at(5, 9, 0), masked_fraction=0.5)
        _a_set_beside(page, 236, complete=False)
        _pick(page.resume_choice, "require")
        window._store_changed()
        assert page.main_words() == "Continue from batch 236", page.main_words()
        forget_asked()
        choose(monkeypatch, words="Cancel")
        page._main_clicked()
        question = all_asked()[0]
        assert (question.title, question.answers, question.default) == (
            "Continue from batch 236 although a check failed?", ("Continue from batch 236 anyway", "Cancel"),
            "Cancel"), question
        assert len(stand_in.calls) == 1
        forget_asked()
        choose(monkeypatch, default=True)
        kept = SimpleNamespace(id="20261005T090000", config={"model": "NADROWSKI", "chi_mode": True}, inputs={},
                               note="")
        assert page.resume(kept) is True, "the Resume did not start"
        _the_call(stand_in, 2)
        assert all_asked() == [], "a Resume asked the pre-run question"
        assert stand_in.calls[-1][1]["resume_fit"] == "20261005T090000"
        stand_in.release(None)
        _settle_the_run()


def test_the_pre_run_checks_read_the_prior_in_use_whatever_posterior_is_held(tmp_path, monkeypatch):
    """With a posterior trained on another prior held beside the prior in use, as a load on the Training step allows,
    "Before you start" and the question before a training read the audit of the prior in use, which the training will
    use, never the audit of the posterior's own prior; the Checks step's rows go on reading the posterior's own. The
    prior in use's audit failed and the other's passed: the line names the failure and the press asks first. The
    reverse: the line says the audit passed, with the prior in use's date, and nothing is asked."""
    from core.gui.design import date_text
    from tests._runs import at
    qt_app()
    other = "20200101T000000"
    with _third_step_of_the_frequency_response(tmp_path, monkeypatch) as (window, screen, page, stand_in):
        _a_budget(page)
        manifest = SimpleNamespace(parents={"prior": other}, body={}, created="2026-10-02T09:00:00+00:00")
        screen.session.posterior = SimpleNamespace(kind="posterior", id="20261002T090000", name="retrain_other",
                                                   manifest=manifest, path=None, posterior=None)
        _an_audit(screen, prior=other, created=at(3, 14, 0))
        _an_audit(screen, created=at(3, 14, 40), masked_fraction=0.5)
        window._store_changed()
        assert _checks_line(page)[1] == ("Attention. Frequency band: Not run yet. Unusable driven recordings: audit: "
                                         f"Failed, {date_text(at(3, 14, 40))}. Starting will ask first.")
        audit = screen.check_rows()[1]
        assert (audit.state, audit.when) == ("Passed", at(3, 14, 0)), "the Checks step read the prior in use's audit"
        forget_asked()
        choose(monkeypatch, default=True)
        page._main_clicked()
        assert [one.title for one in all_asked()] == ["Start training although a check failed?"]
        assert stand_in.calls == []
        _an_audit(screen, prior=other, created=at(4, 9, 0), masked_fraction=0.5)
        _an_audit(screen, created=at(4, 9, 30))
        _a_band(screen, created=at(4, 8, 0))
        window._store_changed()
        assert _checks_line(page)[1] == (f"Fine. Frequency band: Passed, {date_text(at(4, 8, 0))}. Unusable driven "
                                         f"recordings: audit: Passed, {date_text(at(4, 9, 30))}.")
        assert screen.check_rows()[1].state == "Failed", "the Checks step read the prior in use's audit"
        forget_asked()
        choose(monkeypatch, default=True)
        page._main_clicked()
        _the_call(stand_in, 1)
        assert all_asked() == [], "the failed audit of the posterior's own prior asked first"
        stand_in.release(None)
        _settle_the_run()


def test_the_checks_are_read_at_their_moments_and_never_while_a_run_is_live(tmp_path, monkeypatch):
    """What the checks need is read once at each moment: at a store change, by the screen when no page that reads the
    checks is in sight, and then by the Training page alone when it is; at the page's showing after that; at a run's end,
    by the page in sight, or by the screen when the page is not; never while a run is live, whatever is pressed, shown or
    worked out, with a posterior and its simulation set held, nothing of the run's or the checks' read at all; every
    reading asked otherwise is made, two in one turn of the event loop as two; and working the steps' states out reads
    nothing."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import build_run_host, readings_of_a_run, write_simulation_set
    from tests._look_helpers import _settle
    from tests._runs import running, turn_until
    app = qt_app()
    with _third_step(tmp_path, monkeypatch, held=False) as (window, screen, page, stand_in):
        window.go_to("inference", "prior")
        _settle()
        calls = readings_of_a_run(monkeypatch)

        def read() -> int:
            return calls.count("read_records")

        window._store_changed()
        assert read() == 1, "a store change with no page that reads the checks in sight"
        _settle()
        calls.clear()
        window.go_to("inference", "training")
        _settle()
        assert read() == 1, "the Training page's showing, marked to read by the store's change"
        calls.clear()
        window._store_changed()
        _settle()
        assert read() == 1, "a store change with the Training page in sight"
        calls.clear()
        page.form.rows["num_runs"].widget.setText("4321")
        page.read_timer.timeout.emit()
        assert read() == 0, "the reading after typing read the checks"
        calls.clear()
        screen.recompute_states()
        screen.refresh_gates()
        assert read() == 0, "the steps' states worked out read the checks"
        page._main_clicked()
        assert stand_in.started.wait(5)
        folder = write_simulation_set(default_store().kind_dir("simulation"), {"a set of": "the moments"}, 2,
                                      complete=True, chi_masked=[(1, 4), (2, 4)])
        text = tmp_path / "posterior"
        text.mkdir()
        (text / "log.txt").write_text("", encoding="utf-8")
        manifest = SimpleNamespace(parents={"prior": screen.session.inf_prior.id, "simulation": folder.name}, body={},
                                   created="2026-10-03T09:00:00+00:00")
        screen.session.posterior = SimpleNamespace(kind="posterior", id="20261003T090000", name="retrain_1", path=text,
                                                   manifest=manifest, posterior=None)
        calls.clear()
        window._store_changed()
        screen.refresh_gates()
        screen.read_checks()
        window.go_to("inference", "prior")
        _settle()
        window.go_to("inference", "training")
        _settle()
        assert calls == [], f"read while a run is live: {calls}"
        stand_in.release(SimpleNamespace(kind="posterior", id="p1", name="retrain_2", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        _settle()
        assert read() == 1, "the run's end with the Training page in sight"
        window.go_to("inference", "prior")
        _settle()
        calls.clear()
        with running(build_run_host()):
            pass
        _settle()
        assert read() == 1, "a run's end with no page that reads the checks in sight"
        calls.clear()
        screen.read_checks()
        screen.read_checks()
        assert read() == 2, "a reading asked was not made"


def test_the_screen_reads_a_sets_counts_from_its_state_else_from_the_posteriors_text(tmp_path):
    """The one reading of what the checks need reads, for the posterior in use, its simulation set's counts of unusable
    driven recordings from the set's saved state when that holds one pair for each batch done, else the first total in
    the posterior's own text, else none; and asks the engine whether the prior in use is the one the posterior was
    trained with, keeping its sentence; and when the set was completed, its making and the seconds it took. Between
    readings these are kept for the same records, and are not known for another posterior until the next reading. The
    state kept from before the newest, read when the newest cannot be, is never taken for the whole set's counts."""
    import json
    from datetime import datetime, timezone
    import torch
    from torch.distributions import Categorical, MixtureSameFamily, Normal
    from core.artifacts import ArtifactStore, use_store
    from core.gui.check_readings import WHOLE_SET_SCOPES
    from tests._fixtures import write_simulation_set
    from tests._runs import masked_line, scope_part

    def mixture(shift):
        return MixtureSameFamily(Categorical(torch.ones(2)), Normal(torch.tensor([0.0, shift]), torch.ones(2)))

    qt_app()
    store = ArtifactStore(tmp_path / "Artifacts")
    with use_store(store):
        screen = build_inference()
        identity = {"a set of": "the checks"}
        folder = write_simulation_set(store.kind_dir("simulation"), identity, 3, complete=True,
                                      chi_masked=[(1, 4), (2, 4), (3, 4)])
        text = tmp_path / "posterior"
        text.mkdir()
        (text / "log.txt").write_text(f"10:00:00 info {masked_line(5, 50, 2, scope_part(7))}\n"
                                      f"10:00:01 info {masked_line(1, 2, 1, WHOLE_SET_SCOPES[1])}\n", encoding="utf-8")
        manifest = SimpleNamespace(parents={"prior": "20261001T222935", "simulation": folder.name},
                                   body={"transform": {"V": [[1.0]]}}, created="2026-10-03T09:00:00+00:00")
        posterior = SimpleNamespace(kind="posterior", id="20261003T031200", name="retrain", path=text,
                                    manifest=manifest, posterior=SimpleNamespace(latent=SimpleNamespace(
                                        prior=mixture(1.0)), truncation=None))
        prior = SimpleNamespace(kind="prior", id="20261001T222935", name="retrain_prior", prior=mixture(1.0),
                                manifest=None)
        (folder / "manifest.json").write_text(json.dumps({"created": "2026-10-03T08:00:00+00:00",
                                                          "body": {"wall_seconds": 3600}}), encoding="utf-8")
        screen.session = SbiSession(inf_prior=prior, posterior=posterior)
        screen.read_checks()
        held = screen.checks_held.held
        assert (held.set_counts, held.set_done, held.pair_refused) == ((6, 12, 3, WHOLE_SET_SCOPES[0]), True, ""), held
        assert held.set_when == datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc), "the set's making taken as its end"
        assert screen.check_rows()[2].state == "Waiting for the audit"
        screen.session.posterior = SimpleNamespace(**{**posterior.__dict__, "id": "20261003T031201"})
        assert screen.check_rows()[2].line == ("Not compared: the counts of unusable driven recordings could not be "
                                              "read."), "another posterior's counts were taken as read"
        screen.session.posterior = posterior
        assert screen.check_rows()[2].state == "Waiting for the audit", "the counts read were not kept"
        prior.prior = mixture(2.0)
        torch.save({"batches_done": 3, "complete": True, "rng": None, "chi_masked": [[1, 4], [2, 4]]},
                   folder / "state.pt")
        screen.read_checks()
        assert screen.checks_held.held.set_counts == (5, 50, 2, scope_part(7)), (
            "counts for fewer batches than the set holds were taken for the whole set")
        write_simulation_set(store.kind_dir("simulation"), identity, 3, complete=True)
        screen.read_checks()
        held = screen.checks_held.held
        assert held.set_counts == (5, 50, 2, scope_part(7)), "the posterior's text was not read for the first total"
        assert held.pair_refused.startswith("The calibration: the prior supplied is not the one this posterior was "
                                            "trained with"), held.pair_refused
        (text / "log.txt").unlink()
        screen.read_checks()
        assert screen.checks_held.held.set_counts is None
        (text / "log.txt").write_text(f"10:00:00 info {masked_line(5, 50, 2, scope_part(7))}\n", encoding="utf-8")
        (folder / "state.pt").write_bytes(b"torn")
        torch.save({"batches_done": 2, "complete": False, "rng": None, "chi_masked": [[1, 4], [2, 4]]},
                   folder / "state.prev.pt")
        screen.read_checks()
        assert screen.checks_held.held.set_counts == (5, 50, 2, scope_part(7)), (
            "the counts of the state kept from before the newest were taken for the whole set")


def test_the_checks_list_reads_once_and_never_while_a_run_is_live(tmp_path, monkeypatch):
    """The Checks list reads what the checks need once at each moment, by its own reading, the screen leaving it to the
    page while the page is in sight: at its first showing, at a store change, on "Check again", at a run's end; a check
    opened or the page drawn again reads nothing; and while a run is live nothing of the checks, the run history or a
    simulation set is read, whatever is shown, opened, pressed or worked out, the page showing what it read last."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.run.controller import controller
    from tests._fixtures import build_window, readings_of_a_run
    from tests._look_helpers import _settle
    from tests._runs import Clock, StandIn, at, controller_on, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        shown_off_screen(window, 1300, 820)
        window.go_to("inference", "prior")
        _settle()
        calls = readings_of_a_run(monkeypatch)
        page = window.part("inference").validate_panel

        def read() -> int:
            return calls.count("read_records")

        window.go_to("inference", "checks")
        _settle()
        assert read() == 1, "the list's first showing"
        calls.clear()
        page.open_check("calibration")
        page.open_check(None)
        page.refresh_local_gates()
        window.part("inference").refresh_gates()
        assert read() == 0, "opening a check or drawing the page read the checks"
        window._store_changed()
        assert read() == 1, "a store change with the list in sight"
        calls.clear()
        page.foot.check_again.click()
        assert read() == 1, "Check again"
        calls.clear()
        stand_in = StandIn(word="Inference")
        start_run(window.part("inference").tsnpe_panel, stand_in)
        window.go_to("inference", "prior")
        _settle()
        window.go_to("inference", "checks")
        _settle()
        shown = page.listing.whole.view.model().rowCount()
        page.open_check("audit")
        page.foot.check_again.click()
        page.read_facts()
        page.stale_store()
        window._store_changed()
        window.part("inference").read_checks()
        page.refresh_local_gates()
        _settle()
        assert calls == [], f"read while a run is live: {calls}"
        assert page.listing.whole.view.model().rowCount() == shown
        stand_in.release()
        turn_until(app, lambda: not controller().is_live())
        _settle()
        assert read() == 1, "the run's end with the list in sight"
        window.hide()


# ── the fifth step, Inference ──
def _a_posterior(cfg, prior_id, *, narrowed_around=None):
    """A posterior held, as a stand-in that says what Inference reads of one: of ``cfg``'s kind of observation and
    twelve slots, trained with the prior ``prior_id``; narrowed around the observation ``narrowed_around`` when it is
    handed."""
    return SimpleNamespace(
        kind="posterior", id="20261003T031200", name="retrain_posterior",
        manifest=SimpleNamespace(parents={"prior": prior_id, "observation": narrowed_around},
                                 body={"mode": cfg.observation_mode, "conditioning": {"chi_k_pad": 12}}, config={},
                                 created="2026-10-03T03:12:00+00:00"),
        posterior=SimpleNamespace(x_obs_digest="d" * 16 if narrowed_around else None, truncation=None))


@contextlib.contextmanager
def _fifth_step(tmp_path, monkeypatch, *, chi=False, narrowed=False, own_prior=True, posterior=True):
    """Inference in a whole window in sight, off the screen, on the test's own clock and store: the first step applied;
    a real Nadrowski configuration of the master bounds file, of the frequency response with ``chi``, and a prior of it
    in use; a posterior of it held unless ``posterior`` is False, narrowed around the observation cell_a with
    ``narrowed``, trained with that prior unless ``own_prior`` is False; the page read, and both inference stages
    stand-in runs: (window, screen, page, the two stand-ins by the mode that starts each)."""
    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store, use_store
    from tests._fixtures import _nad_cfg, _prior_artifact
    from tests._runs import StandIn
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window, screen, first = _shown_first_step(tmp_path)
        _apply(first)
        cfg = _nad_cfg(chi_mode=chi)
        prior = default_store().load_prior(cfg, _prior_artifact(default_store(), cfg, name="retrain_prior").id)
        around = _an_observation("cell_a") if narrowed else None
        held = _a_posterior(cfg, prior.id if own_prior else "20260101T000000", narrowed_around=around)
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior,
                                    posterior=held if posterior else None)
        screen.refresh_gates()
        window.go_to("inference", "infer")
        page = screen.infer_panel
        runs = {"simulated": StandIn(word="Inference"), "recordings": StandIn(word="Inference")}
        monkeypatch.setattr(orchestrator, "simulated_inference", runs["simulated"])
        monkeypatch.setattr(orchestrator, "experimental_inference", runs["recordings"])
        page.read_facts()
        yield window, screen, page, runs


def _shown(page) -> set:
    """The keys of the rows Inference stands on the page, but the names and the note."""
    return {key for key, row in page.form.rows.items() if row.is_shown()} - {"name", "observation_name", "note"}


def _ready(page, tmp_path, *, mode) -> None:
    """Every row of ``mode`` given: a cell file of the model, or the recordings the posterior reads, a length and the
    drive; the observation named."""
    rows = page.form.rows
    page.modes.set_value(mode)
    rows["t_obs"].widget.setText("4.5")
    page.names.other_name.setText("cell_b")
    if mode == "simulated":
        page.cell_picker.restore_key("master_spont.txt")
        return
    for stem in ("spont.npy", "forced.npy", "driven.npy"):
        (tmp_path / stem).touch()
    rows["recording_spont"].widget.edit.setText(str(tmp_path / "spont.npy"))
    if page.kind() == "forced":
        rows["recording_forced"].widget.edit.setText(str(tmp_path / "forced.npy"))
        for key, text in (("drive_amplitude", "1.5"), ("drive_frequency", "5"), ("drive_phase", "0"),
                          ("drive_offset", "0")):
            rows[key].widget.setText(text)
    else:
        rows["chi_f0_si"].widget.setText("1.0")
        row = page.driven.rows()[0]
        row.path.edit.setText(str(tmp_path / "driven.npy"))
        row.hz.setText("12.5")


def _made(name="cell_b") -> tuple:
    """What an inference's run returns, as stand-ins that say what a page reads of them: the observation ``name`` and
    the inference made on it."""
    return (SimpleNamespace(kind="observation", id="20261113T140000", name=name, manifest=None, path=None),
            SimpleNamespace(kind="inference", id="20261113T140001", name="", manifest=None, path=None, results={}))


def _run_ends(stand_in, payload=None) -> None:
    """The stand-in run ``stand_in`` returns ``payload`` (an inference's two records, by default), and has ended."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    stand_in.release(_made() if payload is None else payload)
    turn_until(qt_app(), lambda: not controller().is_live(), 10.0)


def test_the_two_modes_never_mix_and_the_second_shows_the_posteriors_form(tmp_path, monkeypatch):
    """The two modes, both named at the top of the form: a simulated cell shows its rows alone, and the recordings of
    the posterior in use show theirs alone, named on the page: one driven recording and its drive, or a frequency
    response's table and its drive amplitude. What one mode was given is never passed by the other: a simulated cell's
    call holds no recording, and the recordings' call no cell."""
    from core import orchestrator
    for chi, rows, reads in ((False, {"recording_spont", "recording_forced", "drive_amplitude", "drive_frequency",
                                      "drive_phase", "drive_offset"}, "This posterior reads one driven recording."),
                             (True, {"recording_spont", "recording_probe", "chi_f0_si"},
                              "This posterior reads a frequency response.")):
        (tmp_path / str(chi)).mkdir()
        with _fifth_step(tmp_path / str(chi), monkeypatch, chi=chi) as (_window, _screen, page, runs):
            assert [button.text() for button in page.modes.buttons.values()] == ["A simulated cell",
                                                                                 "Your own recordings"]
            assert page.form.rows["infer_mode"].label_text() == "Run on"
            assert page.mode() == "simulated" and _shown(page) == {"infer_mode", "cell", "cell_typed", "t_obs",
                                                                     "n_samples"}
            assert page.reads_line.isHidden()
            _ready(page, tmp_path, mode="simulated")
            _ready(page, tmp_path, mode="recordings")
            assert _shown(page) == {"infer_mode", "t_obs", "n_samples", *rows}, (chi, _shown(page))
            assert page.reads_line.plain_text() == reads
            page._main_clicked()
            assert runs["recordings"].started.wait(5), (chi, page.foot.reason.text())
            args, kwargs = runs["recordings"].calls[-1]
            assert not {"cell", "gt_values"} & set(kwargs) and args[2].spont.endswith("spont.npy"), kwargs
            _run_ends(runs["recordings"])
            page.modes.set_value("simulated")
            page.names.other_name.setText("cell_c")      # the run's end emptied the name its records now hold
            page.read_facts()
            page._main_clicked()
            assert runs["simulated"].started.wait(5), (chi, page.foot.reason.text())
            args, kwargs = runs["simulated"].calls[-1]
            assert kwargs["cell"].endswith("master_spont.txt") and not any(
                "npy" in str(value) for value in kwargs.values()), kwargs
            _run_ends(runs["simulated"])
            assert orchestrator.simulated_inference is runs["simulated"]


def test_with_no_posterior_the_second_mode_says_why_in_place_of_a_form(tmp_path, monkeypatch):
    """With no posterior in use, "Your own recordings" shows no form, since the form is the one the posterior reads: the
    page says so in its place, and the start is greyed with the reason, a step a person can take."""
    with _fifth_step(tmp_path, monkeypatch, posterior=False) as (_window, _screen, page, _runs):
        page.modes.set_value("recordings")
        page.redraw()
        assert _shown(page) == {"infer_mode", "n_samples"}, _shown(page)
        assert page.reads_line.plain_text() == "The form depends on the posterior. Load or train one in step 3."
        said = [(fact.kind, fact.sentence) for fact in page.facts_from_form()]
        assert ("lock", "Needs a posterior: train one in step 3, or load a saved one.") in said, said
        assert page.foot.reason.text().startswith("Cannot start yet: needs a posterior"), page.foot.reason.text()
        assert not page.foot.main_button.isEnabled()


def test_the_recording_length_and_drive_amplitude_open_blank_at_every_launch(tmp_path, monkeypatch):
    """The recording length and the drive amplitude open blank at every launch, and so does every row of the drive: an
    inference started with them writes none of them, and the keys an older page wrote of them are read by nothing."""
    from core.gui import settings as st
    with _fifth_step(tmp_path, monkeypatch, chi=True) as (_window, _screen, page, runs):
        _ready(page, tmp_path, mode="recordings")
        page._main_clicked()
        assert runs["recordings"].started.wait(5)
        _run_ends(runs["recordings"])
    held = st.settings()
    written = [key for key in held.allKeys() if key.startswith("inference_infer/")]
    assert written and not [key for key in written if "tobs" in key or "t_obs" in key or "f0" in key], written
    for old in ("sim_tobs", "exp_tobs", "chi_tobs", "chi_f0_si"):
        held.setValue(f"inference_infer/{old}", "7.5")
    held.sync()
    again = build_inference().infer_panel
    blank = ("t_obs", "chi_f0_si", "drive_amplitude", "drive_frequency", "drive_phase")
    assert [again.form.rows[key].widget.text() for key in blank] == [""] * len(blank)


def test_the_observation_must_be_named_and_the_inference_need_not_be(tmp_path, monkeypatch):
    """The observation an inference makes needs a name, as every record but an inference does: left blank it is said
    and greys the start; the inference's own name may stay blank. A name another observation holds is said under its
    row in the window's words once it has been read."""
    from tests._fixtures import qt_app as _qt_app
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, _runs):
        _ready(page, tmp_path, mode="simulated")
        page.names.other_name.setText("")
        page.redraw()
        said = [(fact.kind, fact.sentence, fact.blocks) for fact in page.facts_from_form()]
        assert ("err", "Type a name for the observation.", "a name is needed") in said, said
        assert page.names.name_value() == "" and not any("inference" in sentence for _kind, sentence, _ in said)
        assert page.form.rows["name"].label_text() == "Name of the inference"
        assert page.form.rows["observation_name"].label_text() == "Name of the observation"
        _an_observation("cell_a")
        page.names.other_name.setText("cell_a")
        page.names.check_now()
        _qt_app().processEvents()
        line = page.form.rows["observation_name"].line
        assert (line.kind, line.words) == ("err", "The name 'cell_a' is taken by another observation. Choose another "
                                                  "name."), (line.kind, line.words)
        page.names.other_name.setText("cell_b")
        page.names.check_now()
        page.redraw()
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()


@pytest.mark.parametrize("mode", ["simulated", "recordings"])
def test_a_narrowed_posterior_asks_at_every_run_in_both_modes(mode, tmp_path, monkeypatch):
    """A posterior narrowed around one observation asks before every inference, in both modes, the question's words
    exact for each, "Cancel" preselected: "Cancel" starts nothing; "Run inference anyway" hands the run the consent to
    run on another observation; and the next press asks again, for the answer is never kept."""
    from core.artifacts import Accept
    from core.gui.step_facts import EXTRAPOLATION_TEXT, SAME_RECORDINGS_TEXT, SIMULATED_NEVER_TEXT
    with _fifth_step(tmp_path, monkeypatch, narrowed=True) as (_window, _screen, page, runs):
        _ready(page, tmp_path, mode=mode)
        forget_asked()
        page._main_clicked()
        said = asked()
        beneath = f"{SIMULATED_NEVER_TEXT if mode == 'simulated' else SAME_RECORDINGS_TEXT} {EXTRAPOLATION_TEXT}"
        assert (said.title, said.sentence, said.beneath, said.answers, said.default) == (
            "Run on another observation?",
            "This posterior is valid only near the observation it was narrowed around, cell_a.", beneath,
            ("Run inference anyway", "Cancel"), "Cancel"), said
        assert not runs[mode].started.is_set(), "Cancel started a run"
        choose(monkeypatch, words="Run inference anyway")
        page._main_clicked()
        assert runs[mode].started.wait(5)
        assert runs[mode].calls[-1][1]["accept"] == Accept(other_observation=True)
        _run_ends(runs[mode])
        page.names.other_name.setText("cell_c")          # the run's end emptied the name its records now hold
        page.read_facts()
        forget_asked()
        choose(monkeypatch)
        page._main_clicked()
        assert [box.title for box in all_asked()] == ["Run on another observation?"], "the answer was kept"
        assert len(runs[mode].calls) == 1


def test_the_inference_page_reads_the_model_in_use_alone(tmp_path, monkeypatch):
    """With a Hopf configuration in use beside Nadrowski's cells and a Nadrowski cell remembered, the list of cells is
    Hopf's and the remembered cell is not chosen; the drive's rows are Hopf's, its second amplitude among them. Back to
    Nadrowski, its own again: its cells, the cell remembered for it, and its drive."""
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    from core.gui.remembered import CELL_KEY, CELL_MODEL_KEY
    from core.gui import settings as st
    held = st.settings()
    held.setValue(CELL_KEY, "master_spont.txt")
    held.setValue(CELL_MODEL_KEY, "NADROWSKI")
    held.sync()
    hopf = cli.make_sim_config("HOPF", VALID_LABELS[VALID_MODELS.index("HOPF")], registry.state_dep_drift("HOPF"),
                               str(BOUNDS_PATH / "hopf" / "cell.txt"))
    with _fifth_step(tmp_path, monkeypatch) as (_window, screen, page, _runs):
        nadrowski = screen.session.cfg
        assert (page.cell_picker.base_path.name, page.cell_picker.key()) == ("nadrowski", "master_spont.txt")
        for cfg, model, cells, chosen, drives in (
                (hopf, "hopf", ["cell.txt"], "", {"drive_amplitude", "drive_amp_y", "drive_frequency", "drive_phase",
                                                  "drive_offset"}),
                (nadrowski, "nadrowski", None, "master_spont.txt", {"drive_amplitude", "drive_frequency",
                                                                    "drive_phase", "drive_offset"})):
            screen.session.draft.model = cfg.model
            screen.session.cfg = cfg
            screen.session.posterior = _a_posterior(cfg, screen.session.inf_prior.id)
            screen.refresh_gates()
            page.read_facts()
            combo = page.cell_picker.combo
            listed = [combo.itemData(at) for at in range(combo.count())]
            assert page.cell_picker.base_path.name == model and (cells is None or listed == cells), (model, listed)
            assert page.cell_picker.key() == chosen, (model, page.cell_picker.key())
            page.modes.set_value("recordings")
            shown = {key for key in _shown(page) if key.startswith("drive_")}
            assert shown == drives, (model, shown)
            page.modes.set_value("simulated")


def test_the_remembered_rows_are_written_when_an_inference_starts_and_never_at_closing(tmp_path, monkeypatch):
    """What an inference started with is written once its run has started, each key at that moment and at no other: the
    mode, a simulated cell's file with its model, the recordings' paths under the kind they are read as and the driven
    recordings, how many and each one; nothing as the form is filled in or a choice is made, and nothing at closing."""
    from PySide6.QtCore import QSettings
    written = []
    kept = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda qs, key, value: (
        written.append(f"{qs.group()}/{key}" if qs.group() else key), kept(qs, key, value))[1])
    plans = ["plans/last_used/nadrowski", "plans/last_used_at/nadrowski"]
    for chi in (False, True):
        (tmp_path / str(chi)).mkdir()
        with _fifth_step(tmp_path / str(chi), monkeypatch, chi=chi) as (window, _screen, page, runs):
            written.clear()
            _ready(page, tmp_path, mode="simulated")
            page.read_facts()
            assert written == [], "a choice of Inference was written as it was made"
            page._main_clicked()
            assert runs["simulated"].started.wait(5)
            assert sorted(set(written)) == ["inference_infer/cell", "inference_infer/cell_model",
                                            "inference_infer/infer_mode", *plans], written
            _run_ends(runs["simulated"])
            written.clear()
            _ready(page, tmp_path, mode="recordings")
            page._main_clicked()
            assert runs["recordings"].started.wait(5)
            recordings = (["inference_infer/chi_spont", "inference_infer/driven/0/file",
                           "inference_infer/driven/0/frequency", "inference_infer/driven/size"] if chi else
                          ["inference_infer/exp_forced", "inference_infer/exp_spont"])
            assert sorted(set(written)) == sorted(["inference_infer/infer_mode", *recordings, *plans]), written
            _run_ends(runs["recordings"])
            written.clear()
            window._save_state()
            assert not [key for key in written if key.startswith("inference_infer/")], written


def test_a_remembered_cell_of_another_model_is_not_selected(tmp_path, monkeypatch):
    """A cell file is remembered with the model it is a cell of, and chosen again only for that model: one remembered
    for Hopf is not chosen in Nadrowski's list, though Nadrowski's folder holds a file of that name."""
    from core.gui.remembered import CELL_KEY, CELL_MODEL_KEY
    from core.gui import settings as st
    held = st.settings()
    held.setValue(CELL_KEY, "master_spont.txt")
    held.setValue(CELL_MODEL_KEY, "HOPF")
    held.sync()
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, _runs):
        combo = page.cell_picker.combo
        assert "master_spont.txt" in [combo.itemData(at) for at in range(combo.count())]
        assert page.cell_picker.key() == "", "a cell remembered for another model was chosen"


def test_the_cell_check_runs_in_the_reading_and_never_while_a_run_is_live(tmp_path, monkeypatch):
    """The cell chosen is checked against the configuration in use by the page's reading alone: not as it is chosen,
    but 300 ms after, or at the next reading; while a run is live, never, the line saying so, and at the run's end,
    once."""
    from core import cli
    from core.gui.run.controller import controller
    from tests._runs import StandIn, start_run, turn_until
    checked = []
    real = cli.validate_gt_file
    monkeypatch.setattr(cli, "validate_gt_file", lambda cfg, path: (checked.append(path), real(cfg, path))[1])
    with _fifth_step(tmp_path, monkeypatch) as (_window, screen, page, _runs):
        checked.clear()
        page.cell_picker.restore_key("master_spont.txt")
        assert checked == [], "the cell was checked as it was chosen"
        page.read_timer.timeout.emit()
        assert [path.endswith("master_spont.txt") for path in checked] == [True], checked
        stand_in = StandIn(word="Other run")
        start_run(screen.prior_panel, stand_in)
        page.cell_picker.restore_key("master_weak.txt")
        page.read_timer.timeout.emit()
        page.read_facts()
        page.foot.check_again.click()
        assert len(checked) == 1, "the cell was checked while a run was live"
        stand_in.release()
        turn_until(qt_app(), lambda: not controller().is_live(), 10.0)
        assert len(checked) == 2 and checked[-1].endswith("master_weak.txt"), checked


def _a_record(folder, figures, **fields):
    """A record an inference's run returns, as a stand-in that says what the page reads of it: ``fields``, and its figures
    by their files' names, each a small picture in ``folder``."""
    from PySide6.QtGui import QImage
    folder.mkdir(parents=True, exist_ok=True)
    for name in figures:
        image = QImage(8, 6, QImage.Format_RGB32)
        image.fill(0xFFFFFF)
        image.save(str(folder / name))
    return SimpleNamespace(manifest=SimpleNamespace(figures=list(figures)), path=folder, **fields)


#: An inference's estimates, as its record holds them: in the record's order, an assumed input among them.
SUMMARY = [{"name": "k", "q05": 0.11, "median": 0.2345, "q95": 0.4, "assumed": False},
           {"name": "x_scale", "q05": 20.0, "median": 30.5, "q95": 41.0, "assumed": False},
           {"name": "t_scale", "q05": 2.5, "median": 3.1, "q95": 4.0, "assumed": False},
           {"name": "T", "q05": 290.0, "median": 300.0, "q95": 310.0, "assumed": True}]


def _inferred(page, tmp_path, *, mode="simulated", coverage=0.93, observation="cell_b", saved_as="cell_b", truth=True,
              digest=None, summary=SUMMARY):
    """Run an inference on ``page`` in ``mode`` to its end, its stand-in run returning an observation saved as
    ``saved_as`` (asked ``observation``), and an inference saved as retrain_inf whose record holds ``summary``, the
    truth when ``truth``, ``coverage`` and three figures; the observation's record holds its trace, and the observation
    the digest ``digest``. The observation's id."""
    from core import orchestrator
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    _ready(page, tmp_path, mode=mode)
    page.names.other_name.setText(observation)
    page.names.name.setText("retrain_inf")
    page.read_facts()
    stand_in = orchestrator.simulated_inference if mode == "simulated" else orchestrator.experimental_inference
    page._main_clicked()
    assert stand_in.started.wait(5), page.foot.reason.text()
    made = _an_observation(saved_as or "unnamed_made_here")        # the observation the stage saves
    held = _a_record(tmp_path / "observation", ["simulated_observation.png"], kind="observation", id=made,
                     name=saved_as, digest=digest)
    results = {"posterior_summary": summary, "ppc": {"coverage_90": coverage},
               "ground_truth": {"k": 0.2, "x_scale": 31.0, "t_scale": 3.0, "T": 300.0} if truth else None}
    inference = _a_record(tmp_path / "inference", ["posterior_predictive_check.png", "posterior_corner.png",
                                                   "eye_test.png"],
                          kind="inference", id="20261113T150000", name="retrain_inf", results=results)
    stand_in.release((held, inference))
    turn_until(qt_app(), lambda: not controller().is_live(), 10.0)
    return made


def test_the_result_names_each_parameters_values_with_units_and_the_true_value(tmp_path, monkeypatch):
    """After a simulated cell's inference the page says its result, read from the two records once the run is over: a
    ruled table of each parameter's 5 %, 50 % and 95 % values in the record's order, an assumed input marked so, the
    unit of each from the configuration in use, and the value the cell was simulated with; the corner figure first, then
    the inference's other figures and the observation's trace, each captioned by its title; how much of the observation
    the posterior predicts; and what was saved. The run's own figures stand on its run's page alone. The observation
    comes into use."""
    with _fifth_step(tmp_path, monkeypatch) as (_window, screen, page, _runs):
        _inferred(page, tmp_path)
        table = page.result.table_words()
        assert table.columns == ("Parameter", "5 %", "50 %", "95 %", "Unit", "True value")
        assert table.rows == (("k", "0.11", "0.2345", "0.4", "", "0.2"),
                              ("x_scale", "20", "30.5", "41", "nm", "31"),
                              ("t_scale", "2.5", "3.1", "4", "ms", "3"),
                              ("T (assumed input)", "290", "300", "310", "", "300")), table.rows
        assert table.right == frozenset({1, 2, 3, 5})
        assert [caption.text() for caption, _picture in page.result.pictures] == [
            "Posterior corner", "Posterior predictive check", "Eye test", "Simulated observation"]
        assert [(fact.kind, fact.sentence) for fact in page.result.sentences()] == [
            ("info", "Of the observation's features, 93 % fall inside what the posterior predicts (90 % interval)."),
            ("info", "Saved as the inference retrain_inf and the observation cell_b.")]
        assert [step.words for step in page.result.next_steps()] == ["Narrow around this observation",
                                                                    "Run another inference"]
        assert not page.result.isHidden() and page.figure_stack.isHidden()
        assert getattr(screen.session.observation, "name", None) == "cell_b"


#: A parameter named by one long word, which a result's table keeps whole: the table scrolls sideways in its block at
#: the window's least size.
_LONG_NAMED = {"name": "the_ratio_of_the_bundles_stiffness_to_its_drag_at_the_lowest_frequency_the_drive_reaches",
               "q05": 1.0, "median": 2.0, "q95": 3.0, "assumed": False}


@pytest.mark.parametrize("factor", [1.0, 1.5])
@pytest.mark.parametrize("wide", [False, True], ids=["fits", "scrolls sideways"])
def test_the_results_table_block_is_a_stop_of_the_tab_key_where_the_page_is_read(wide, factor, tmp_path, monkeypatch):
    """At the window's least size, at both text sizes, the block the result's table stands in, made after the page set
    the Tab key's order, has its place in that order where the page is read, before the result's next steps and the
    foot, whether its table fits it or scrolls sideways in it; while it scrolls, the Tab key stops on it there."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui import design
    from tests._look_helpers import _settle, text_size
    from tests._window_helpers import in_front
    with text_size(factor), _fifth_step(tmp_path, monkeypatch) as (window, _screen, page, _runs):
        _inferred(page, tmp_path, summary=[*SUMMARY, _LONG_NAMED] if wide else SUMMARY)
        window.resize(design.scaled(900), design.scaled(600))
        _settle()
        block, after = page.result.table_block, page.result.next_buttons[0]
        assert block.scrolls() is wide, (factor, "the table does not stand as this case says")
        chain, at = [], page.help_button.nextInFocusChain()
        while at is not page.help_button and len(chain) < 2000:
            chain.append(at)
            at = at.nextInFocusChain()
        foot = [page.foot.facts.scroller, *page.foot.stops()]
        assert chain.index(block) < chain.index(after) < min(chain.index(stop) for stop in foot), (
            "the table's block stands out of the page's order")
        walked = []
        with in_front(window):
            page.help_button.setFocus(Qt.TabFocusReason)
            for _ in range(80):
                QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
                _settle()
                if QApplication.focusWidget() in walked:
                    break
                walked.append(QApplication.focusWidget())
        reached = [walked.index(stop) for stop in foot if stop in walked]
        assert (block in walked) is wide and after in walked and reached, [type(stop).__name__ for stop in walked]
        assert (not wide or walked.index(block) < walked.index(after)) and walked.index(after) < min(reached), (
            factor, [type(stop).__name__ for stop in walked])


def test_the_coverage_sentence_is_left_out_when_the_record_holds_none(tmp_path, monkeypatch):
    """A record that holds no share of the observation's features says nothing of it; recordings carry no truth, so the
    table has no column for it; and an observation saved without the name it was asked, another record having taken
    that name while the run went, is said so."""
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, _runs):
        _inferred(page, tmp_path, mode="recordings", coverage=None, observation="cell_b", saved_as="", truth=False)
        assert page.result.table_words().columns == ("Parameter", "5 %", "50 %", "95 %", "Unit")
        said = [(fact.kind, fact.sentence) for fact in page.result.sentences()]
        assert not any("fall inside" in sentence for _kind, sentence in said), said
        assert said[-1] == ("warn", "It was saved without its name: 'cell_b' was taken while the run was going."), said
        assert said[0][1].startswith("Saved as the inference retrain_inf and the observation Unnamed observation"), said


def test_narrow_around_this_observation_opens_the_round(tmp_path, monkeypatch):
    """"Narrow around this observation" opens the narrowing round with the observation just made chosen in its list,
    another observation listed before it; nothing is applied and no run starts. "Run another inference" stays on the
    page, its result, its names and its note emptied."""
    from core.gui.run.controller import controller
    with _fifth_step(tmp_path, monkeypatch) as (window, screen, page, runs):
        _an_observation("cell_a")
        made = _inferred(page, tmp_path)
        held = (screen.session.cfg, screen.session.inf_prior, screen.session.posterior, screen.session.observation)
        calls = len(runs["simulated"].calls)
        page.result.next_buttons[0].click()
        assert window.current_place() == ("inference", "narrowing")
        assert screen.tsnpe_panel.obs_picker.key() == made, "the observation just made is not chosen on the round"
        assert (screen.session.cfg, screen.session.inf_prior, screen.session.posterior,
                screen.session.observation) == held, "the round's opening changed what is in use"
        assert not controller().is_live() and len(runs["simulated"].calls) == calls
        window.go_to("inference", "infer")
        page.names.note.setText("a note")
        page.result.next_buttons[1].click()
        assert window.current_place() == ("inference", "infer") and page.result.isHidden()
        assert (page.names.name_value(), page.names.other_value(), page.names.note_value()) == ("", "", "")


@pytest.mark.parametrize("narrowed, digest, offered", [
    (True, "e" * 16, ("Run another inference",)),
    (True, "d" * 16, ("Narrow around this observation", "Run another inference")),
    (False, "e" * 16, ("Narrow around this observation", "Run another inference"))],
    ids=["narrowed around another observation", "narrowed around this observation", "not narrowed"])
def test_narrow_around_this_observation_is_offered_only_where_the_round_can_use_it(narrowed, digest, offered, tmp_path,
                                                                                     monkeypatch):
    """An inference's result offers "Narrow around this observation" only where the narrowing round can be run around
    that observation, by the rule the round greys its list of observations by: with a posterior no narrowing round made,
    or one narrowed around this same observation, by its digest; after an inference with a posterior narrowed around
    another, "Run another inference" alone."""
    with _fifth_step(tmp_path, monkeypatch, narrowed=narrowed) as (_window, _screen, page, _runs):
        choose(monkeypatch, words="Run inference anyway")
        _inferred(page, tmp_path, digest=digest)
        assert tuple(step.words for step in page.result.next_steps()) == offered, page.result.next_steps()


def test_an_inference_that_returns_its_records_empties_the_names_and_the_note_it_was_handed(tmp_path, monkeypatch):
    """When an inference returns its records, the names (the inference's and the observation's) and the note its run was
    handed are emptied, as the Training step's are at its run's end, so that the next inference is not stopped by a name
    its own records now hold; a field whose words changed while the run went keeps what was typed since. The result
    stays on the page until a next step is pressed."""
    from core import orchestrator
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, _runs):
        _ready(page, tmp_path, mode="simulated")
        page.names.other_name.setText("cell_b")
        page.names.name.setText("retrain_inf")
        page.names.note.setText("a note")
        page.read_facts()
        stand_in = orchestrator.simulated_inference
        page._main_clicked()
        assert stand_in.started.wait(5), page.foot.reason.text()
        page.names.note.setText("another note")                    # typed while the run went
        made = _an_observation("cell_b")                            # the observation the stage saves
        held = _a_record(tmp_path / "observation", ["simulated_observation.png"], kind="observation", id=made,
                         name="cell_b")
        inference = _a_record(tmp_path / "inference", ["posterior_corner.png"], kind="inference",
                              id="20261113T150000", name="retrain_inf",
                              results={"posterior_summary": SUMMARY, "ppc": {"coverage_90": 0.93}})
        stand_in.release((held, inference))
        turn_until(qt_app(), lambda: not controller().is_live(), 10.0)
        assert (page.names.name_value(), page.names.other_value(), page.names.note.text()) == (
            "", "", "another note"), "the names the run was handed still stand, or the note typed since went"
        assert not page.result.isHidden() and page.result.next_steps()[-1].words == "Run another inference"
        page.read_facts()
        assert "taken" not in page.foot.reason.text(), page.foot.reason.text()


@pytest.mark.parametrize("end", ["stopped", "failed", "stopped, another name typed since"])
def test_an_inference_that_ends_once_its_observation_is_saved_empties_that_name(end, tmp_path, monkeypatch):
    """An inference stopped, or failed, once its observation was saved under the name it was handed: the observation's
    name is emptied, since that observation holds it now and the next press would be refused for it, unless another
    name was typed there while the run went, which stands; the inference's name and the note, which no record holds,
    stand as typed."""
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, runs):
        _ready(page, tmp_path, mode="simulated")                    # the observation named cell_b
        page.names.name.setText("retrain_inf")
        page.names.note.setText("a note")
        page.read_facts()
        page._main_clicked()
        assert runs["simulated"].started.wait(5), page.foot.reason.text()
        if end.endswith("typed since"):
            page.names.other_name.setText("cell_c")
        _an_observation("cell_b")                                   # the observation the stage saved
        if end == "failed":
            runs["simulated"].fail(RuntimeError("the sampler found no finite value"))
        else:
            controller().request_stop()
        turn_until(qt_app(), lambda: not controller().is_live(), 10.0)
        assert (page.names.other_value(), page.names.name_value(), page.names.note.text()) == (
            "cell_c" if end.endswith("typed since") else "", "retrain_inf", "a note"), (
            "the observation's name, now taken, still stands, or another name went")
        page.read_facts()
        assert "taken" not in page.foot.reason.text(), page.foot.reason.text()


def test_an_inference_name_taken_meanwhile_empties_the_observations_name_and_says_why(tmp_path, monkeypatch):
    """An inference whose own name another inference took while the run went is refused once its observation is saved:
    the run's page says the refusal, and Inference empties the observation's name, which that observation now holds, and
    says under what is in use that the observation was saved and the inference needs another name."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg
    from tests._runs import turn_until
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, runs):
        _ready(page, tmp_path, mode="simulated")
        page.names.name.setText("retrain_inf")
        page.read_facts()
        page._main_clicked()
        assert runs["simulated"].started.wait(5), page.foot.reason.text()
        _an_observation("cell_b")                                   # the observation the stage saved
        with default_store().create("inference", _nad_cfg(), name="retrain_inf") as writer:
            writer.body = {"results": {}}                           # the inference another program saved meanwhile
        runs["simulated"].fail(Refusal("an inference named 'retrain_inf' already exists; rename or delete it first",
                                       field="name"))
        turn_until(qt_app(), lambda: not controller().is_live(), 10.0)
        assert page.names.other_value() == "", "the observation's name, now taken, still stands"
        said = page.in_use.said
        assert not said.isHidden() and said.sentence.plain_text() == (
            "Attention. The observation was saved as cell_b; give the inference another name."), (
            said.sentence.plain_text())
        assert page.names.name_value() == "retrain_inf"


# ── the sixth step, the narrowing round ──
@contextlib.contextmanager
def _sixth_step(tmp_path, monkeypatch, *, narrowed=False, network=(256, 10), chi=False):
    """The narrowing round in a whole window in sight, off the screen, on the test's own clock and store: the first step
    applied; a real Nadrowski configuration of the master bounds file and a prior of it in use; two observations,
    cell_a and cell_b, of other digests; a posterior trained with that prior at ``network`` held, narrowed around cell_a
    with ``narrowed``; of the frequency response with ``chi``; the observation loaded at the press a stand-in of the one
    chosen; the round's stage a stand-in run; the page read: (window, screen, page, the stand-in run, the observations'
    ids by name)."""
    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store, use_store
    from tests._fixtures import _nad_cfg, _prior_artifact, write_observation
    from tests._runs import StandIn
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        window, screen, first = _shown_first_step(tmp_path)
        _apply(first)
        cfg = _nad_cfg(chi_mode=True) if chi else _nad_cfg()
        prior = default_store().load_prior(cfg, _prior_artifact(default_store(), cfg, name="retrain_prior").id)
        made = {name: write_observation(default_store(), cfg, name=name, digest=mark * 16)
                for name, mark in (("cell_a", "a"), ("cell_b", "b"))}
        posterior = SimpleNamespace(
            kind="posterior", id="20261003T031200", name="retrain_posterior",
            manifest=SimpleNamespace(parents={"prior": prior.id, "observation": made["cell_a"] if narrowed else None},
                                     body={"mode": cfg.observation_mode, "amortized": not narrowed,
                                           "training": {"hidden_features": network[0], "num_transforms": network[1]}},
                                     config={}, created="2026-10-03T03:12:00+00:00"),
            posterior=SimpleNamespace(x_obs_digest="a" * 16 if narrowed else None, truncation=None))
        screen.session = SbiSession(draft=screen.session.draft, cfg=cfg, inf_prior=prior, posterior=posterior)
        screen.refresh_gates()
        window.go_to("inference", "narrowing")
        page = screen.tsnpe_panel
        run = StandIn(word="Narrowing round")
        monkeypatch.setattr(orchestrator, "tsnpe_round", run)
        monkeypatch.setattr(ArtifactStore, "load_observation", lambda store, cfg, ref: SimpleNamespace(
            kind="observation", id=ref, name=store.get("observation", ref).name))
        page.read_facts()
        yield window, screen, page, run, made


def _ready_round(page, made, name="retrain_round1", observation="cell_a") -> None:
    page.obs_picker.restore_key(made[observation])
    page.names.name.setText(name)
    page.read_facts()


def _round_ends(run, payload=None) -> None:
    from core.gui.run.controller import controller
    from tests._runs import turn_until
    run.release(payload if payload is not None else _round_posterior())
    turn_until(qt_app(), lambda: not controller().is_live(), 10.0)


def _round_posterior(acceptance=0.12, ident="20261113T160000"):
    """What a round around cell_a returns, as a stand-in that says what the pages read of a posterior from a narrowing
    round."""
    return SimpleNamespace(kind="posterior", id=ident, name="retrain_round1",
                           manifest=SimpleNamespace(parents={}, body={"amortized": False, "training": {
                               "tsnpe_acceptance": acceptance}}, config={}, created="2026-11-13T16:00:00+00:00"),
                           posterior=SimpleNamespace(x_obs_digest="a" * 16, truncation=None))


@pytest.mark.parametrize("prior", ["in Records", "no longer in Records"])
def test_the_round_asks_for_the_posteriors_own_prior_only_while_it_can_be_loaded(prior, tmp_path, monkeypatch):
    """A posterior in use trained with a prior other than the one in use blocks the round: while that prior is in
    Records, the round names it and asks for it to be loaded in step 2; once it is no longer there, a sentence of its
    own says so and asks for nothing that cannot be loaded."""
    from core.artifacts import default_store
    from tests._fixtures import _nad_cfg, _prior_artifact
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, _made):
        older = _prior_artifact(default_store(), _nad_cfg(), name="older_prior", seed=5).id
        screen.session.posterior.manifest.parents["prior"] = older
        if prior == "no longer in Records":
            default_store().delete("prior", older, force=True)
        page.read_facts()
        said = [(fact.kind, fact.sentence) for fact in page.facts_from_form()]
        want = {"in Records": "This posterior was trained with older_prior; the prior in use is retrain_prior. Load "
                              "older_prior in step 2.",
                "no longer in Records": "This posterior was trained with a prior no longer in Records; the prior in "
                                        "use is retrain_prior. A narrowing round refuses this pair."}[prior]
        assert ("err", want) in said, said
        assert page.foot.reason.text().startswith("Cannot start yet:") and not page.foot.main_button.isEnabled()


def test_the_round_passes_the_network_it_shows(tmp_path, monkeypatch):
    """The round hands its stage the network the page shows, the width and the depth always, at their usual values as
    typed ones, so that what is on the page is what trains; and the region's level and directions, the budget and the
    fit's settings, each as an argument; the epoch limit only when one is typed."""
    from tests._runs import turn_until
    with _sixth_step(tmp_path, monkeypatch) as (_window, _screen, page, run, made):
        _ready_round(page, made)
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        args, kwargs = run.calls[-1]
        assert args[3].id == made["cell_a"], args
        handed = {key: kwargs.get(key) for key in ("hidden_features", "num_transforms", "level", "n_directions",
                                                   "num_runs", "run_size_cap", "learning_rate", "stop_after_epochs",
                                                   "checkpoint_every", "resume", "new_run", "if_name_taken", "name")}
        assert handed == {"hidden_features": 128, "num_transforms": 8, "level": 0.999, "n_directions": 5,
                          "num_runs": 5000, "run_size_cap": 0, "learning_rate": 0.001, "stop_after_epochs": 20,
                          "checkpoint_every": 50, "resume": "auto", "new_run": False, "if_name_taken": "unnamed",
                          "name": "retrain_round1"}, handed
        assert "max_num_epochs" not in kwargs
        _round_ends(run)
        page.form.rows["hidden_features"].widget.setText("64")
        page.form.rows["num_transforms"].widget.setText("4")
        page.form.rows["max_num_epochs"].widget.set_value(30)
        page.names.name.setText("retrain_round2")
        count = len(run.calls)
        page._main_clicked()
        turn_until(qt_app(), lambda: len(run.calls) > count, 5.0)
        _args, kwargs = run.calls[-1]
        assert (kwargs["hidden_features"], kwargs["num_transforms"], kwargs["max_num_epochs"]) == (64, 4, 30)
        _round_ends(run)


@pytest.mark.parametrize("step", ["Inference", "the narrowing round"])
def test_no_widget_an_inferences_or_a_rounds_run_end_makes_is_shown_as_a_window_of_its_own(step, tmp_path,
                                                                                           monkeypatch):
    """STANDING, beside the builders' guard and the Checks page's. What Inference makes from its run's press to the
    run's end, its result read from an inference's records with its estimates and its figures, and what the narrowing
    round makes from its press to its run's end, its result read from the round's posterior, is made inside the page:
    none is shown as a window of its own but the window itself and the questions the dialogs' guard records, what the
    run's end asks the event loop to do later included."""
    from tests._look_helpers import _settle
    from tests._window_helpers import windows_of_their_own
    if step == "Inference":
        with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, _runs):
            with windows_of_their_own(monkeypatch) as shown:
                _inferred(page, tmp_path)
                _settle()                                    # what the run's end asked for later has had its turn
            made = (page.result.table_words() is not None, len(page.result.pictures) > 0)
    else:
        with _sixth_step(tmp_path, monkeypatch) as (_window, _screen, page, run, made):
            _ready_round(page, made)
            with windows_of_their_own(monkeypatch) as shown:
                page._main_clicked()
                assert run.started.wait(5), page.foot.reason.text()
                _round_ends(run)
                _settle()                                    # what the run's end asked for later has had its turn
            made = (len(page.result.sentences()) > 0, len(page.result.next_steps()) == 2)
    assert made == (True, True), f"the run's end made no result: nothing is tested ({made})"
    assert shown == [], f"{len(shown)} widgets were shown as windows of their own: {shown}"


def test_the_network_of_the_posterior_in_use_is_said_and_marked_when_it_differs(tmp_path, monkeypatch):
    """Under the network: the posterior in use's own, from its record, and marked "Attention." under the depth while the
    round trains another."""
    with _sixth_step(tmp_path, monkeypatch, network=(256, 10)) as (_window, _screen, page, _run, _made):
        page.redraw()
        line = page.form.rows["num_transforms"].line
        assert (line.kind, line.words) == ("warn", "The posterior in use was trained at 256 by 10; this round trains "
                                                   "128 by 8."), (line.kind, line.words)
        page.form.rows["hidden_features"].widget.setText("256")
        page.form.rows["num_transforms"].widget.setText("10")
        page.redraw()
        assert not page.form.rows["num_transforms"].line.says()
        assert page.network_line.plain_text() == "The posterior in use was trained at 256 by 10."


def test_a_region_level_below_the_engines_warning_is_said_before_the_run(tmp_path, monkeypatch):
    """A region level below the engine's warning is said under it before the run, and greys nothing: narrowing that
    tight removes part of the prior for good. The page's threshold is the engine's default less its margin."""
    from core.gui.step_facts import REGION_WARN_BELOW
    from core.SBI import truncate
    assert REGION_WARN_BELOW == round(truncate.DEFAULT_HPD - 0.009, 6)
    with _sixth_step(tmp_path, monkeypatch) as (_window, _screen, page, _run, made):
        _ready_round(page, made)
        row = page.form.rows["hpd_level"]
        row.widget.setText("0.95")
        page.redraw()
        assert (row.line.kind, row.line.words) == ("warn", "Tighter than 0.99. Narrowing removes part of the prior for "
                                                           "good; no later round gets it back.")
        assert page.foot.main_button.isEnabled(), page.foot.reason.text()
        row.widget.setText("0.995")
        page.redraw()
        assert not row.line.says()


def test_the_rounds_shared_rows_read_as_trainings_and_are_held_to_the_same_rules(tmp_path, monkeypatch):
    """The eight settings a round shares with a training are named as the Training page names them and held to the same
    rules, from the one table both pages read: a value out of rule says the same line under its row on both pages."""
    from core.gui.panels.inference.posterior_tab import PosteriorPanel
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from core.gui.step_facts import FIT_RULES
    from tests._fixtures import code_only
    shared = ("num_runs", "run_size_cap", "hidden_features", "num_transforms", "learning_rate", "stop_after_epochs",
              "max_num_epochs", "checkpoint_every")
    assert set(shared) <= set(FIT_RULES)
    assert "fit_row_problem" in code_only(PosteriorPanel._row_problem) and "fit_row_problem" in code_only(
        TSNPEPanel.problems)
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, _made):
        training = screen.posterior_panel
        for key in shared:
            for one in (page, training):
                widget = one.form.rows[key].widget
                if hasattr(widget, "ticked_box"):
                    widget.ticked_box.setChecked(False)
                (widget.field if hasattr(widget, "field") else widget).setText("-1" if key != "learning_rate" else "0")
                one.redraw()
            lines = [(one.form.rows[key].label_text(), one.form.rows[key].line.words) for one in (page, training)]
            assert lines[0] == lines[1] and lines[0][1], (key, lines)


def test_a_new_observation_moves_the_list_and_says_so(tmp_path, monkeypatch):
    """An observation an inference makes moves the round's list to it once that run has ended, never while it is live;
    the line under the list says which it replaced, through every later reading, and goes once the person picks
    another observation."""
    from tests._runs import StandIn, start_run
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, made):
        page.obs_picker.restore_key(made["cell_a"])
        stand_in = StandIn(word="Inference")
        start_run(screen.infer_panel, stand_in)
        screen.take_observation(SimpleNamespace(kind="observation", id=made["cell_b"], name="cell_b", manifest=None))
        page.redraw()
        assert page.obs_picker.key() == made["cell_a"], "the list moved while a run was live"
        _round_ends(stand_in, payload=object())
        assert page.obs_picker.key() == made["cell_b"]
        line = page.form.rows["observation"].line
        assert (line.kind, line.words) == ("warn", "Changed to the observation just made, cell_b. Before: cell_a.")
        page.read_facts()
        assert page.form.rows["observation"].line.words.startswith("Changed to"), "a reading took the line away"
        page.obs_picker.combo.setCurrentIndex(page.obs_picker.combo.findData(made["cell_a"]))
        page.redraw()
        assert not page.form.rows["observation"].line.says(), "the line outlived the person's own pick"


def test_the_observations_a_narrowed_posterior_cannot_use_are_greyed_with_their_reason(tmp_path, monkeypatch):
    """With a posterior narrowed around cell_a in use, cell_b, of another digest, is listed greyed with why, and chosen
    all the same it is said under the list and greys the start; cell_a stays choosable, with the list's own tip, so one
    is on offer. The rows are read through the list's own data, no item of it held."""
    from PySide6.QtCore import Qt
    with _sixth_step(tmp_path, monkeypatch, narrowed=True) as (_window, _screen, page, _run, made):
        combo = page.obs_picker.combo
        at_b, at_a = combo.findData(made["cell_b"]), combo.findData(made["cell_a"])

        def choosable(at):
            return bool(combo.model().flags(combo.model().index(at, 0)) & Qt.ItemFlag.ItemIsEnabled)

        assert not choosable(at_b) and combo.itemData(at_b, Qt.ItemDataRole.ToolTipRole) == (
            "Cannot be used: this posterior was narrowed around another observation.")
        assert choosable(at_a) and combo.itemData(at_a, Qt.ItemDataRole.ToolTipRole).startswith(made["cell_a"]), (
            "the list's own tip of a usable row was taken away")
        assert page.observation_on_offer()
        combo.setCurrentIndex(at_b)
        page.names.name.setText("retrain_round1")
        page.redraw()
        line = page.form.rows["observation"].line
        assert (line.kind, line.words) == ("err", "Cannot be used: this posterior was narrowed around another "
                                                  "observation.")
        assert page.foot.reason.text().startswith("Cannot start yet: the observation cannot be used")


def test_the_step_six_mark_follows_the_rounds_held_reading(tmp_path, monkeypatch):
    """Step 6 can start only while its list holds an observation the posterior in use can be drawn around, as the
    round's own reading found them: with a posterior narrowed around an observation none listed is, it cannot."""
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, _made):
        assert page.observation_on_offer() and screen.step_states()[5].can_start
        screen.session.posterior.posterior.x_obs_digest = "f" * 16
        page.read_facts()
        screen.recompute_states()
        assert not page.observation_on_offer() and not screen.step_states()[5].can_start


def test_narrow_around_this_observation_selects_it_on_the_round(tmp_path, monkeypatch):
    """A next step that carries an observation opens the round with that observation chosen in its list, the step shown
    as it takes it; nothing is asked, nothing applied, and no run starts."""
    from core.gui.run.controller import controller
    with _sixth_step(tmp_path, monkeypatch) as (window, screen, page, run, made):
        page.obs_picker.restore_key(made["cell_a"])
        window.go_to("inference", "infer")
        forget_asked()
        held = (screen.session.cfg, screen.session.inf_prior, screen.session.posterior, screen.session.observation)
        screen.open_with("narrowing", {"observation": made["cell_b"]})
        assert window.current_place() == ("inference", "narrowing") and page.obs_picker.key() == made["cell_b"]
        assert all_asked() == [] and not controller().is_live() and run.calls == []
        assert (screen.session.cfg, screen.session.inf_prior, screen.session.posterior,
                screen.session.observation) == held


def test_a_new_posterior_keeps_the_observation_and_a_new_prior_lets_it_go(tmp_path, monkeypatch):
    """The observation is the posterior's data, not its product: a posterior that comes into use keeps the observation in
    use, and step 5 stays done; a prior that comes into use lets it go, with the posterior."""
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, _page, _run, made):
        observation = SimpleNamespace(kind="observation", id=made["cell_a"], name="cell_a", manifest=None)
        screen.take_observation(observation)
        screen.take_posterior(_round_posterior())
        assert screen.session.observation is observation and screen.step_states()[4].done
        screen.take_prior(screen.session.cfg, screen.session.inf_prior)
        assert screen.session.observation is None and screen.session.posterior is None


def test_the_round_reads_the_model_in_use_alone(tmp_path, monkeypatch):
    """With a Hopf configuration in use beside Nadrowski's observations none of them is listed, an observation of
    another model's width being of no use to it; back to Nadrowski, they are listed again."""
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    hopf = cli.make_sim_config("HOPF", VALID_LABELS[VALID_MODELS.index("HOPF")], registry.state_dep_drift("HOPF"),
                               str(BOUNDS_PATH / "hopf" / "cell.txt"))
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, made):
        nadrowski = screen.session.cfg
        screen.session.cfg = hopf
        screen.refresh_gates()
        page.read_facts()
        assert page.obs_picker.combo.count() == 0 and not page.observation_on_offer()
        screen.session.cfg = nadrowski
        screen.refresh_gates()
        page.read_facts()
        assert {page.obs_picker.combo.itemData(at) for at in range(page.obs_picker.combo.count())} == set(made.values())


def test_the_rounds_run_hands_its_history_plain_settings(tmp_path, monkeypatch):
    """The round's run hands its history what it was started with as names and numbers alone: the model and its options,
    the bounds file, the prior, the posterior it is drawn from and the observation by their ids, the region's level and
    directions, the name, the note and each fit's setting."""
    import json
    from core.gui.run.controller import controller
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, run, made):
        _ready_round(page, made)
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        held = controller().model.spec.settings
        json.dumps(held)
        assert {key: held[key] for key in ("prior", "parent_posterior", "observation", "level", "directions", "name")} == {
            "prior": screen.session.inf_prior.id, "parent_posterior": "20261003T031200", "observation": made["cell_a"],
            "level": 0.999, "directions": 5, "name": "retrain_round1"}, held
        # what a Continue after a reopening fills step 1 and step 2 from: the model, its options and the bounds file
        assert (held["model"], held["bounds"], sorted(held["options"])) == ("NADROWSKI", "master.txt", [
            "chi_k_pad", "chi_max_cycles", "chi_mode", "chi_n_freqs", "reparam_rotate", "units_override"]), held
        assert [row["name"] for row in held["rows"]] == ["num_runs", "run_size_cap", "hidden_features",
                                                          "num_transforms", "learning_rate", "stop_after_epochs",
                                                          "max_num_epochs", "checkpoint_every"]
        _round_ends(run)


def test_the_round_rereads_its_lists_at_run_ended_and_never_while_live(tmp_path, monkeypatch):
    """While a run is live the round reads nothing: neither the observations' digests nor anything a run may write; at
    its end, in sight, it reads once."""
    from core.gui.panels.inference import tsnpe_tab
    from tests._fixtures import readings_of_a_run
    from tests._runs import StandIn, start_run
    with _sixth_step(tmp_path, monkeypatch, narrowed=True) as (_window, screen, page, _run, _made):
        read = []
        real = tsnpe_tab.observation_digests
        monkeypatch.setattr(tsnpe_tab, "observation_digests", lambda store: (read.append(1), real(store))[1])
        stand_in = StandIn(word="Other run")
        start_run(screen.prior_panel, stand_in)
        calls = readings_of_a_run(monkeypatch)
        page.read_facts()
        page.foot.check_again.click()
        page.redraw()
        screen.refresh_gates()
        assert (read, calls) == ([], []), (read, calls)
        _round_ends(stand_in, payload=object())
        assert len(read) == 1, read


def test_the_controller_publishes_the_idle_state_before_the_run_ends(tmp_path, monkeypatch):
    """STANDING. As a run ends the controller says first that no run is live, then that the run has ended: a page's
    reading at the run's end, as the round's is, finds no run live. ``status_changed`` with a status not live comes
    before ``run_ended``, and a listener of ``run_ended`` reads ``is_live()`` False."""
    from core.gui.run.controller import controller
    from tests._runs import StandIn, start_run
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, _page, _run, _made):
        heard = []
        held = controller()

        def status(status):
            heard.append(("status", status.live))

        def ended(_status):
            heard.append(("ended", held.is_live()))

        held.status_changed.connect(status)
        held.run_ended.connect(ended)
        try:
            stand_in = StandIn(word="Other run")
            start_run(screen.prior_panel, stand_in)
            _round_ends(stand_in, payload=object())
        finally:                             # the process's one run controller is left with the listeners it had
            held.status_changed.disconnect(status)
            held.run_ended.disconnect(ended)
        at = heard.index(("ended", False))
        assert heard[at - 1] == ("status", False), heard


def test_the_round_says_its_result_and_offers_check_this_posterior(tmp_path, monkeypatch):
    """Once its run has ended the round says its posterior is in use from step 3 on, narrowed around its observation,
    and how much of the prior's draws its region kept; that the posterior it came from stays in Records; and offers to
    check the new posterior or run an inference on the same observation. The round's posterior is in use."""
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, run, made):
        _ready_round(page, made)
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        result = _round_posterior(acceptance=0.12)
        _round_ends(run, result)
        assert [(fact.kind, fact.sentence) for fact in page.result.sentences()] == [
            ("info", "In use from step 3 on: retrain_round1, narrowed around cell_a. The region kept 12 % of the "
                     "prior's draws."),
            ("info", "The posterior it came from, retrain_posterior, stays in Records; load it in step 3 to go back.")]
        assert [(step.words, step.step) for step in page.result.next_steps()] == [
            ("Check this posterior", "checks"), ("Run inference on the same observation", "infer")]
        assert screen.session.posterior is result


def test_the_training_steps_list_moves_to_the_rounds_posterior_after_the_round_ends(tmp_path, monkeypatch):
    """The posterior a round returns is what step 3 holds once the round's run has ended: the Training page, not in sight,
    reads nothing while the run is live nor at its end, and at its next show reads once, its list of saved posteriors
    standing on the round's posterior."""
    from core.artifacts import default_store
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _posterior_artifact
    with _sixth_step(tmp_path, monkeypatch) as (window, screen, page, run, made):
        training = screen.posterior_panel
        saved = _posterior_artifact(default_store(), _nad_cfg(), name="round_saved").id
        read = []
        real = training.read_disk
        monkeypatch.setattr(training, "read_disk", lambda: (read.append(controller().is_live()), real())[1])
        _ready_round(page, made)
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        _round_ends(run, _round_posterior(ident=saved))
        assert read == [], read
        window.go_to("inference", "training")
        assert read == [False] and training.post_picker.key() == saved, (read, training.post_picker.key())


def test_a_rounds_expected_epochs_are_said_as_the_last_narrowing_round_of_its_kind(tmp_path):
    """The epochs a round is expected to take come from the newest finished narrowing round of its kind, and its words
    say so, never "the last training"."""
    import json
    from core.artifacts.store import MANIFEST, Summary
    from core.gui.run import controller as controller_module
    path = tmp_path / "Artifacts" / "posteriors" / "retrain_round1__x"
    path.mkdir(parents=True)
    (path / MANIFEST).write_text(json.dumps({"config": {"model": "NADROWSKI"}, "body": {
        "mode": "chi", "training": {"epochs_trained": 40}}}), encoding="utf-8")
    listed = [Summary("posterior", "retrain_round1", "retrain_round1", "2026-11-05T10:00:00+00:00", "", path, True,
                      None, mode="chi", amortized=False)]
    store = SimpleNamespace(list=lambda kind: listed if kind == "posterior" else [])
    assert controller_module._expected(store, {"model": "NADROWSKI", "observation": "chi", "narrowed": True}) == (
        40, "about 40 epochs: the last narrowing round of this kind, retrain_round1, ran 40")
    assert controller_module._expected(store, {"model": "NADROWSKI", "observation": "chi", "narrowed": False}) is None


def test_a_failed_pre_run_check_asks_before_the_round_loads_or_starts_anything(tmp_path, monkeypatch):
    """After the audit of the prior in use read "Failed", "Start the narrowing round" asks one question first, "Cancel"
    preselected, before the observation is loaded or anything is dispatched: "Cancel" starts nothing; "Start the
    narrowing round anyway" loads the observation and dispatches the round once."""
    from core.artifacts import ArtifactStore
    from tests._runs import at
    with _sixth_step(tmp_path, monkeypatch, chi=True) as (window, screen, page, run, made):
        loads = []
        held = ArtifactStore.load_observation
        monkeypatch.setattr(ArtifactStore, "load_observation",
                            lambda store, cfg, ref: (loads.append(ref), held(store, cfg, ref))[1])
        _an_audit(screen, created=at(3, 14, 40), masked_fraction=0.5)
        window._store_changed()
        _ready_round(page, made)
        forget_asked()
        choose(monkeypatch, default=True)
        page._main_clicked()
        question = asked()
        assert (question.title, question.answers, question.default) == (
            "Start the narrowing round although a check failed?", ("Start the narrowing round anyway", "Cancel"),
            "Cancel"), question
        assert (loads, run.calls, len(all_asked())) == ([], [], 1), "the round loaded or started before its question"
        forget_asked()
        choose(monkeypatch, words="Start the narrowing round anyway")
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        assert (loads, len(run.calls), len(all_asked())) == ([made["cell_a"]], 1, 1), (loads, run.calls)
        _round_ends(run)


def test_a_carried_observation_not_listed_yet_is_held_and_chosen_at_the_next_reading(tmp_path, monkeypatch):
    """While a run is live a next step's observation is not looked for in the store: the page lists nothing, holds the
    id and says under the list that it is chosen once the step reads its list again; when the run has ended the round,
    in sight, reads, chooses it, and the line goes."""
    from core.artifacts import default_store
    from tests._fixtures import write_observation
    from tests._runs import StandIn, start_run
    with _sixth_step(tmp_path, monkeypatch) as (_window, screen, page, _run, made):
        page.obs_picker.restore_key(made["cell_a"])
        stand_in = StandIn(word="Other run")
        start_run(screen.prior_panel, stand_in)
        carried = write_observation(default_store(), screen.session.cfg, name="cell_c", digest="c" * 16)
        listed, real = [], page.obs_picker.refresh
        monkeypatch.setattr(page.obs_picker, "refresh", lambda: (listed.append(1), real())[1])
        page.take_carried({"observation": carried})
        line = page.form.rows["observation"].line
        assert (listed, page.obs_picker.key()) == ([], made["cell_a"]), "the store was listed while a run was live"
        assert (line.kind, line.words) == (
            "info", f"The observation carried here, {carried}, is chosen once this step reads its list again."), (
            line.kind, line.words)
        _round_ends(stand_in, payload=object())
        assert page.obs_picker.key() == carried and not page.form.rows["observation"].line.says()


def test_the_round_lets_go_of_what_it_was_started_with_once_it_is_over(tmp_path, monkeypatch):
    """The objects a round was started with are kept only for the question the stage's refusal of a new simulation set
    asks: let go at the end of a run that was not refused before anything was spent, and once that question is
    answered."""
    from core.gui.run.controller import controller
    from core.refusals import Refusal
    from tests._runs import turn_until
    with _sixth_step(tmp_path, monkeypatch) as (_window, _screen, page, run, made):
        _ready_round(page, made)
        page._main_clicked()
        assert run.started.wait(5), page.foot.reason.text()
        assert page._call is not None
        _round_ends(run)
        assert page._call is None, "the objects of a round that ran were kept"
        page.names.name.setText("retrain_round2")
        count = len(run.calls)
        page._main_clicked()
        turn_until(qt_app(), lambda: len(run.calls) > count, 5.0)
        forget_asked()
        choose(monkeypatch, default=True)
        run.fail(Refusal("A saved simulation set one setting away exists: n_runs is 40 there and 5000 here.",
                         field="new_run"))
        turn_until(qt_app(), lambda: not controller().is_live() and all_asked(), 10.0)
        assert all_asked()[-1].title == "Start the narrowing round beside a saved simulation set?"
        assert page._call is None, "the objects were kept after the question was answered"


def test_the_inferences_run_names_the_kind_of_observation_apart_from_any_observation(tmp_path, monkeypatch):
    """An inference's run hands its history the kind of observation it reads under its own key: "observation" names an
    observation's id in every history, a narrowing round's among them."""
    from core.gui.run.controller import controller
    with _fifth_step(tmp_path, monkeypatch) as (_window, _screen, page, runs):
        _ready(page, tmp_path, mode="simulated")
        page._main_clicked()
        assert runs["simulated"].started.wait(5), page.foot.reason.text()
        held = controller().model.spec.settings
        assert held["observation_kind"] == page.kind() and "observation" not in held, held
        _run_ends(runs["simulated"])


def test_the_lengths_the_posterior_was_trained_on_come_from_its_record(tmp_path, monkeypatch):
    """The line under the recording length names the lengths the posterior in use was trained on, from its record, in
    the configuration's unit of time; a record that holds none names the usual range. A length above the longest is
    said to need the posterior to extrapolate, from the same lengths."""
    from core import config
    with _fifth_step(tmp_path, monkeypatch) as (_window, screen, page, _runs):
        per_second = float(screen.session.cfg.get_unit_conversion_factor("s"))
        for record, (low, high) in (({"t_min_exp": 2.0 * per_second, "t_max_exp": 30.0 * per_second}, (2, 30)),
                                    ({}, (config.T_MIN_EXP_S, config.T_MAX_EXP_S))):
            screen.session.posterior.manifest.config = record
            screen.refresh_gates()                  # as a posterior coming into use has it
            page.form.rows["t_obs"].widget.setText(f"{high * 1.5:g}")
            page.read_facts()
            assert page.trained_line.plain_text() == f"The posterior was trained on lengths from {low:g} to {high:g} s."
            said = [row.sentence.plain_text() for row in page.foot.facts.rows()]
            assert [one for one in said if "extrapolate" in one and f"{high:g} s" in one], (record, said)
            page.form.rows["t_obs"].widget.setText(f"{(low + high) / 2:g}")
            page.read_facts()
            assert not [one for one in (row.sentence.plain_text() for row in page.foot.facts.rows())
                        if "extrapolate" in one], record


def test_a_remembered_recording_is_put_back_once_and_never_over_a_box_emptied(tmp_path, monkeypatch):
    """The spontaneous recording remembered for the kind of observation the posterior in use reads is put back once,
    the first time the page knows that kind; a box the person has emptied since stays empty at every later gate."""
    from core.gui import settings as st
    from core.gui.remembered import SPONT_KEYS
    qs = st.settings()
    qs.setValue(SPONT_KEYS["chi"], str(tmp_path / "spont.npy"))
    qs.sync()
    with _fifth_step(tmp_path, monkeypatch, chi=True) as (_window, screen, page, _runs):
        page.modes.set_value("recordings")
        box = page.form.rows["recording_spont"].widget.edit
        assert box.text().endswith("spont.npy"), box.text()
        box.setText("")
        for _ in range(2):
            screen.refresh_gates()
            page.read_facts()
        assert box.text() == "", "a box the person emptied was filled again"


# ── a kept narrowing round ──
def _kept_rounds(store, *, passes=41, name="retrain_round1", **more) -> str:
    """A kept narrowing round in ``store``, its parents a prior, a simulation set, the posterior it narrows and the
    observation it trains around, by ids of their own: its id."""
    from tests._fixtures import build_fit_record, region_dict
    return build_fit_record(store, passes=passes, posterior_name=name, region=region_dict(),
                            parents={"prior": "20261002T101500", "simulation": "7c1e09aa2f31",
                                     "parent_posterior": "20261002T120000", "observation": "20261002T130000"},
                            **more)


def test_the_training_steps_list_holds_no_round(tmp_path, monkeypatch):
    """A kept narrowing round after a reopening is offered on the Narrowing round step alone: the Training step's list
    of kept trainings holds the ordinary ones, and its state says nothing of the round; the round's step says it. The
    table's sentence for a training to resume stays the Training step's."""
    from core.artifacts import ArtifactStore, default_store, use_store
    from core.gui import fields
    from tests._fixtures import build_fit_record, build_window
    from tests._runs import Clock, at, controller_on
    qt_app()
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        ordinary = build_fit_record(default_store(), passes=3, posterior_name="retrain_1")
        kept = _kept_rounds(default_store())
        window = build_window()
        screen = window.part("inference")
        screen.read_offers()
        assert {offer.kept: (offer.kind, offer.step) for offer in screen.offers()} == {
            ordinary: ("resume", "training"), kept: ("round", "narrowing")}
        assert [offer.kept for offer in screen.posterior_panel.kept_offers()] == [ordinary]
        assert [offer.kept for offer, _parts in screen.posterior_panel._kept_rows] == [ordinary]
        assert "narrowing" not in screen.step_states()[2].detail.lower()
        assert fields.CONTROL["resume_fit"] == "Look at the kept trainings in step 3, Training."


def _a_round_page_with_a_kept_round(store, tmp_path, monkeypatch):
    """The window of a launch after one that left a kept narrowing round that can be resumed, the Narrowing round step
    shown: (window, its Parameter inference, the round's page, the round's id)."""
    from tests.test_shell import _a_kept_round_left, _home_of_last_time
    _held, kept = _a_kept_round_left(store, tmp_path, monkeypatch)
    window, _home = _home_of_last_time(tmp_path)
    window.go_to("inference", "narrowing")
    inference = window.part("inference")
    return window, inference, inference.tsnpe_panel, kept


def test_the_kept_rounds_list_reads_once_and_never_while_a_run_is_live(store, tmp_path, monkeypatch):
    """The round page's list of kept rounds is read with the page's reading, once, by the screen's one reading of what
    is offered; drawing it again reads nothing; and while a run is live the page's reading reads none of it, the list
    showing what was read last."""
    from core.SBI import fit_state
    from tests._runs import running
    _window, inference, page, kept = _a_round_page_with_a_kept_round(store, tmp_path, monkeypatch)
    read, real = [], fit_state.kept_fits
    monkeypatch.setattr(fit_state, "kept_fits", lambda *args, **kwargs: read.append(1) or real(*args, **kwargs))
    page.read_facts()
    assert len(read) == 1, read
    page.redraw()
    inference.refresh_gates()
    assert len(read) == 1, "drawing the list read the kept trainings"
    with running(page):
        page.read_facts()
        inference.read_offers()
        assert len(read) == 1, "the kept trainings were read while a run was live"
        assert [offer.kept for offer, _parts in page._kept_rows] == [kept]


def test_the_round_pages_reading_after_the_typing_stops_reads_nothing_kept(store, tmp_path, monkeypatch):
    """The reading the round page makes once the typing stops reads neither what is offered nor what the checks need,
    as the Training page's does: typing makes no record. The kept trainings, the run histories the offers and the
    checks read, and the checks' records are read at a showing of the page, and none of them at a typing pause."""
    from core.gui import check_lineage
    from core.gui.run import history
    from core.SBI import fit_state
    window, inference, page, _kept = _a_round_page_with_a_kept_round(store, tmp_path, monkeypatch)
    read = {}
    for home, name in ((fit_state, "kept_fits"), (history, "runs_of"), (check_lineage, "read_records")):
        def counted(*args, _real=getattr(home, name), _name=name, **kwargs):
            read[_name] = read.get(_name, 0) + 1
            return _real(*args, **kwargs)
        monkeypatch.setattr(home, name, counted)
    page.form.rows["num_runs"].widget.setText("12")
    assert page.isVisible() and page.read_timer.isActive(), "the typing started no reading"
    page.read_timer.stop()
    page.read_timer.timeout.emit()
    assert read == {}, f"the reading after the typing stopped read {read}"
    window.go_to("inference", "training")
    inference.stale_store()
    read.clear()
    window.go_to("inference", "narrowing")
    # the offers' and the checks' histories, once each
    assert read == {"kept_fits": 1, "runs_of": 2, "read_records": 1}, read


def test_a_refused_resume_is_taken_back_at_the_next_reading(store, tmp_path, monkeypatch):
    """A Resume refused before anything is read (here the graphics card gone) says why on the kept round's row, with
    "Delete" alone; a reading after the typing stops leaves it said; once the cause is lifted, the page's next reading
    takes it back and the row offers "Resume" again."""
    from types import SimpleNamespace
    from core import config
    from tests._fixtures import _nad_cfg, region_dict, write_observation
    from tests.test_shell import _a_kept_training, _a_launch_before, _home_of_last_time
    held = _a_launch_before(store, tmp_path, monkeypatch)
    observed = write_observation(store, _nad_cfg(), name="cell_b")
    kept = _a_kept_training(store, held, name="retrain_round1", passes=41, region=region_dict(),
                            config={"device": "cuda"}, parents={"parent_posterior": held.posterior,
                                                                 "observation": observed})
    monkeypatch.setattr(config, "detect_device", config.cpu_device)
    window, _home = _home_of_last_time(tmp_path)
    window.go_to("inference", "narrowing")
    page = window.part("inference").tsnpe_panel
    (_offer, parts), = page._kept_rows
    parts["resume"].click()
    (_offer, parts), = page._kept_rows
    said = "It was trained on the graphics card; this computer has none now."
    assert (page.said_on_row(kept), parts["why"].text()) == (said, said)
    assert not parts["resume"].isVisibleTo(page.kept_block) and parts["delete"].isVisibleTo(page.kept_block)
    page.form.rows["num_runs"].widget.setText("12")
    page.read_timer.stop()
    page.read_timer.timeout.emit()
    assert page.said_on_row(kept) == said, "a reading after the typing stopped took the sentence back"
    monkeypatch.setattr(config, "detect_device", lambda: SimpleNamespace(device=SimpleNamespace(type="cuda")))
    page.read_facts()
    (_offer, parts), = page._kept_rows
    assert page.said_on_row(kept) == "" and not parts["why"].isVisibleTo(page.kept_block)
    assert parts["resume"].isVisibleTo(page.kept_block), "the cause lifted, the row offers no Resume"


def test_a_resume_refusal_is_said_on_the_row_that_was_pressed(store, tmp_path, monkeypatch):
    """A Resume the round refuses for its kept round (a race, a torn record, a process training it) is said on the row
    that was pressed alone, in place of its "Resume", beside its "Delete", the list read again first: no box tells it,
    and nothing is said at the head. Its name taken meanwhile by another process is said there the same way, in the
    window's words, and the next Resume asks for another name. With no row pressed, the box and the head point at the
    kept narrowing rounds on this page; and a Continue refused under the policy that must continue points at the press
    that starts a new round from its first batch. Each sentence under a refusal is the window's own table's."""
    from core.gui import fields
    from core.refusals import Refusal
    from core.SBI import fit_state
    from tests._dialogs import forget_asked
    from tests._fixtures import _nad_cfg, _posterior_artifact
    _window, inference, page, kept = _a_round_page_with_a_kept_round(store, tmp_path, monkeypatch)
    read, real = [], fit_state.kept_fits
    monkeypatch.setattr(fit_state, "kept_fits", lambda *args, **kwargs: read.append(1) or real(*args, **kwargs))
    page._resumed_row = kept
    forget_asked()
    said = "The kept training is being trained by another process."
    page._refusal(Refusal(said, field="resume_fit"))
    (_offer, parts), = page._kept_rows
    assert page.said_on_row(kept) == said and parts["why"].text() == said
    assert not parts["resume"].isVisibleTo(page.kept_block) and parts["delete"].isVisibleTo(page.kept_block)
    assert all_asked() == [] and page._head_fact is None, (all_asked(), page._head_fact)
    assert len(read) == 1, "the list was not read again"
    assert page._resumed_row is None
    # no row pressed: the box, and the head, point at the list on this page
    page.read_facts()
    forget_asked()
    page._refusal(Refusal(said, field="resume_fit"))
    asked, = all_asked()
    fix = "Look at the kept narrowing rounds on this page."
    assert (asked.sentence, asked.beneath) == (said, fix) and fix == fields.fix_sentence("resume_fit", here="narrowing")
    assert page._head_fact.sentence == f"{said} {fix}" and page.said_on_row(kept) == ""
    assert fields.fix_sentence("resume_fit", here="training") == fields.CONTROL["resume_fit"]
    # a Continue refused under the policy that must continue
    page.read_facts()
    forget_asked()
    stage = "resume='require' but there is no resumable cache at the round's folder."
    page._refusal(Refusal(stage, field="resume"))
    asked, = all_asked()
    fix = "Press 'Start the narrowing round' on this page to start a new round from its first batch."
    assert (asked.sentence, asked.beneath) == (stage, fix) and fix == fields.fix_sentence("resume", here="narrowing")
    assert page._head_fact.sentence == f"{stage} {fix}"
    button = page.foot.main_button.text().replace("&", "")
    assert f"Press '{button}' on this page" in fields.fix_sentence("resume", here="narrowing"), (
        "the sentence names the round's main button by other words than the button reads", button)
    # the name taken meanwhile by another process
    page.read_facts()
    _posterior_artifact(store, _nad_cfg(), name="retrain_round1", parents={})
    page._resumed_row, page._resumed_name = kept, "retrain_round1"
    forget_asked()
    page._refusal(Refusal("Another artifact is named retrain_round1.", field="name"))
    (_offer, parts), = page._kept_rows
    taken = "The name 'retrain_round1' is taken by another posterior. Choose another name."
    assert (page.said_on_row(kept), parts["why"].text(), all_asked(), page._head_fact) == (taken, taken, [], None)
    page.read_facts()
    assert inference.resume_round(kept) is None
    asked, = all_asked()
    assert asked.title == "Name this narrowing round", asked.title


def test_a_new_round_over_a_matching_kept_round_asks_first(tmp_path, monkeypatch):
    """A narrowing round started with the settings of a kept one, its prior, the posterior it narrows, its
    observation, its region's level, its budget and its network, asks first, naming the kept round and the epoch it was
    kept after, "Cancel" preselected: "Cancel" starts nothing; "Start a new round" starts it. A round whose settings
    differ asks nothing."""
    from core.artifacts import ArtifactStore
    from core.gui.continuing import Offer
    from tests._dialogs import choose, forget_asked
    from tests._fixtures import build_window
    from tests._runs import Clock, at, controller_on
    from tests.test_pages import _a_round_to_start
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    window = build_window()
    window.go_to("inference", "narrowing")
    screen = window.part("inference")
    page = screen.tsnpe_panel
    _a_round_to_start(page)
    monkeypatch.setattr(ArtifactStore, "load_observation", lambda _store, cfg, ref: SimpleNamespace(
        kind="observation", id=ref, name="cell_a", manifest=None))
    started = []
    monkeypatch.setattr(page, "dispatch", lambda fn, *args, **kwargs: started.append(fn) or False)
    knobs = page._read_knobs()
    _cfg, prior, posterior = page._held()
    match = (prior.id, posterior.id, page.obs_picker.key(), knobs["level"], knobs["num_runs"], knobs["run_size_cap"],
             knobs["hidden_features"], knobs["num_transforms"])
    kept = Offer("round", None, "20261003T130500", None, "", "", epoch=2, label="retrain_round1", step="narrowing",
                 match=match)
    # The press itself, past the reading that counts: the records the round was set up on stood in a store of their own.
    screen.seed_offers((kept,))
    forget_asked()
    page.main_pressed()
    asked, = all_asked()
    assert (asked.title, asked.sentence, asked.answers, asked.default) == (
        "Start a new narrowing round beside a kept one?",
        "The narrowing round retrain_round1 was kept after epoch 2 with these settings. Resume it from the list on this "
        "page to train on from there. A new round starts its fit from the first epoch, leaves the kept one as it is, "
        "and takes its name.", ("Start a new round", "Cancel"), "Cancel")
    assert started == []
    choose(monkeypatch, other=True)
    page.main_pressed()
    assert len(started) == 1
    screen.seed_offers((Offer("round", None, "20261003T130500", None, "", "", epoch=2, label="retrain_round1",
                              step="narrowing", match=(*match[:4], knobs["num_runs"] + 1, *match[5:])),))
    forget_asked()
    page.main_pressed()
    assert all_asked() == [] and len(started) == 2
    # Another name typed: the new round does not take the kept one's, and the question says nothing of it.
    screen.seed_offers((kept,))
    page.names.name.setText("retrain_round9")
    page.names.check_now()
    forget_asked()
    page.main_pressed()
    asked, = all_asked()
    assert asked.sentence == (
        "The narrowing round retrain_round1 was kept after epoch 2 with these settings. Resume it from the list on this "
        "page to train on from there. A new round starts its fit from the first epoch and leaves the kept one as it "
        "is."), asked.sentence


def test_a_kept_round_read_from_its_record_is_matched_by_the_forms_own_settings(store, tmp_path, monkeypatch):
    """A kept narrowing round as its record holds it, read as every kept round is offered, with the prior, the posterior
    it narrows, the observation, the region's level, the batches, the simulations per batch and the network of the
    round the form would start: the press asks first, its question naming the kept round."""
    from core.artifacts import ArtifactStore
    from core.gui import continuing
    from core.SBI import fit_state
    from tests._dialogs import forget_asked
    from tests._fixtures import _nad_cfg, build_window, region_dict
    from tests._runs import Clock, at, controller_on
    from tests.test_pages import _a_round_to_start
    from tests.test_shell import _a_kept_training
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    window = build_window()
    window.go_to("inference", "narrowing")
    screen = window.part("inference")
    page = screen.tsnpe_panel
    _a_round_to_start(page)
    monkeypatch.setattr(ArtifactStore, "load_observation", lambda _store, cfg, ref: SimpleNamespace(
        kind="observation", id=ref, name="cell_a", manifest=None))
    started = []
    monkeypatch.setattr(page, "dispatch", lambda fn, *args, **kwargs: started.append(fn) or False)
    knobs = page._read_knobs()
    _cfg, prior, posterior = page._held()
    ident = _a_kept_training(store, SimpleNamespace(prior=prior.id), name="retrain_round1", passes=2, cfg=_nad_cfg(),
                             region=region_dict(level=knobs["level"]),
                             config={key: knobs[key] for key in ("num_runs", "run_size_cap", "hidden_features",
                                                                 "num_transforms")},
                             parents={"parent_posterior": posterior.id, "observation": page.obs_picker.key()})
    offers = continuing.offers((), (fit_state.kept_fit(store, ident),), records_root=str(store.root),
                               set_batches=lambda _folder: None)
    assert [(offer.kind, offer.why_not) for offer in offers] == [("round", "")], offers
    screen.seed_offers(offers)
    forget_asked()
    page.main_pressed()
    asked, = all_asked()
    assert asked.title == "Start a new narrowing round beside a kept one?" and started == [], (asked, started)


def test_the_rails_step_six_opens_the_kept_rounds_row(store, tmp_path, monkeypatch):
    """With a kept narrowing round offered, the rail's step 6 reads "Unfinished: epoch 41", and opening it from the rail,
    by a click or by the keyboard, shows the round's page with the kept round's row whole in sight, the page scrolled
    away from it before, and the keyboard on its "Resume", resuming nothing."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from core.gui.run.controller import controller
    from tests._window_helpers import in_front
    from tests.test_shell import _a_kept_round_left, _home_of_last_time, _scrolled_away, _whole_in_view
    _held, kept = _a_kept_round_left(store, tmp_path, monkeypatch)
    window, _home = _home_of_last_time(tmp_path)
    inference = window.part("inference")
    inference.read_offers()
    assert inference.step_states()[5].rail_words == "Unfinished: epoch 41"
    page = inference.tsnpe_panel
    with in_front(window):
        _scrolled_away(window, page)
        window.shell.rail.opened.emit("inference", "narrowing")
        for _ in range(5):
            QApplication.processEvents()
        (offer, parts), = page._kept_rows
        assert inference.current_step() == "narrowing" and offer.kept == kept
        assert QApplication.focusWidget() is parts["resume"], QApplication.focusWidget()
        assert _whole_in_view(page, parts["row"]), "the kept round's row is not in sight"
        _scrolled_away(window, page)
        row = window.shell.rail.row("narrowing")
        row.setFocus(Qt.OtherFocusReason)
        QTest.keyClick(row, Qt.Key_Return)
        for _ in range(5):
            QApplication.processEvents()
        (offer, parts), = page._kept_rows
        assert inference.current_step() == "narrowing" and offer.kept == kept
        assert QApplication.focusWidget() is parts["resume"], QApplication.focusWidget()
        assert _whole_in_view(page, parts["row"]), "the kept round's row is not in sight"
    assert not controller().is_live() and inference.session.draft is None
