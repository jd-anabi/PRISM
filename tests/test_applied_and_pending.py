"""What a step holds against what its page shows: "In use" and "Changed, not applied", the rail and Home that follow
them, the question asked before what a session holds is let go, the one table of a page's rules, and what is remembered
of what was applied. Read on the first step, Model and options, on sessions of stand-ins and of the window's own; and
the scan that keeps the shared question the only one a rebuilt page asks."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.gui import chain, record_words
from core.gui.session import SbiSession
from core.gui.step_facts import NOT_HELD, SelectionRow, units_words
from tests._dialogs import all_asked, asked, choose, forget_asked
from tests._fixtures import build_inference, build_window, qt_app

#: The pages of Parameter inference built on the step page, by their files beside it.
REBUILT = ("config_tab.py",)


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
    page.units_text.setText("nm ms pN Hz")
    assert _line(page, "units") == ("", "")


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
    _pick(page.model_combo, "NADROWSKI")
    assert rows["model"].origin.text() == "" and rows["chi_mode"].origin.text() == "kept from last time"
    _apply(page)
    assert rows["chi_mode"].origin.text() == "kept from last time" and rows["model"].origin.text() == ""
    _pick(page.kind_choice, False)
    _pick(page.kind_choice, True)
    assert rows["chi_mode"].origin.text() == "", "a kind touched and put back is still said to be kept"


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
    assert written[1:] == [[("inference_config/model", "BP")]], written
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


def test_an_apply_by_the_keyboard_hands_it_to_the_next_step(tmp_path):
    """The main button greys as it applies, and the keyboard goes on to the way to the next step, not to the rail, where
    the toolkit would hand it from a greyed button. A press by the mouse moves no keyboard."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from tests._look_helpers import _arrive, _settle
    from tests._window_helpers import in_front
    window, screen, page = _shown_first_step(tmp_path)
    with in_front(window):
        page.foot.main_button.setFocus(Qt.TabFocusReason)
        _arrive(page.foot.main_button, Qt.TabFocusReason)
        QTest.keyClick(page.foot.main_button, Qt.Key_Return)
        _settle()
        assert screen.session.draft is not None and QApplication.focusWidget() is page.next_button
        _pick(page.rot_choice, False)
        page.form.rows["model"].widget.setFocus(Qt.TabFocusReason)
        page.foot.main_button.click()
        _settle()
        assert screen.session.draft.reparam_rotate is False and QApplication.focusWidget() is page.model_combo
