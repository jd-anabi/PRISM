"""The rules and tables with no widget: the window's places and keys, the six steps' states, Home's one next step
and first-time test, and what a record is called; and the six steps as the built window holds them.

Torch-free at its module level: the modules under test load neither the toolkit (the record words aside) nor the
engine, and a fresh interpreter proves it, since this process already holds both. A session is a plain
namespace of the five fields the window's session holds. A test that reads the store's own tables or writes a
record imports them inside itself, and so does a test that builds the window.
"""
import ast
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.gui import chain, keys, record_words

REPO = Path(__file__).resolve().parents[1]

# what the rules and tables may never load, and what the record words may never load
_RULES_FORBIDDEN = ("PySide6", "torch", "core.config")
_WORDS_FORBIDDEN = ("torch", "core.config", "core.registry", "core.gui.fields", "core.Helpers.file_manager")

_FRESH_RULES = """
import importlib
import sys
from types import SimpleNamespace as held
for name in sys.argv[2].split(","):
    try:
        importlib.import_module(name)
    except ImportError:
        pass
from core.gui import chain, keys
some = held(draft=held(model="NADROWSKI"), cfg=None, inf_prior=None, posterior=None, observation=None)
every = held(draft=held(model="SHM"), cfg=None, inf_prior=held(name="p"), observation=held(name="o"),
             posterior=held(name="q", posterior=held(truncation=1)))
for session, options in ((some, dict(live="prior", ended=3, pending=frozenset({"model"}))), (every, {}), (None, {})):
    states = chain.step_states(session, observation_on_offer=True, **options)
    said = [state.spoken for state in states]
    chain.next_step(session, states)
    chain.next_step(session, states, live_place=("temperature", "sweep"))
    chain.first_time(has_records=False, has_runs=False, session=session)
said = [chain.model_label("HOPF"), keys.places_words(("training", "narrowing")), keys.place_words("inference/infer")]
early = held(draft=held(model=""), cfg=None, inf_prior=held(name="p"), posterior=None, observation=None)
chain.step_states(early, ended=9500, label=str)
whole = held(draft=held(model="BP"), cfg=None, inf_prior=held(name="p"), observation=held(name="o"),
             posterior=held(name="q", posterior=None))
chain.next_step(whole, chain.step_states(whole, observation_on_offer=True))
said += [keys.place_words("records")]
for lookup, place in ((keys.place_words, "nowhere"), (keys.place_words, "records/prior"), (keys.step_number, "home")):
    try:
        lookup(place)
    except KeyError:
        pass
roots = sys.argv[1].split(",")
bad = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots))
sys.exit(repr(bad) if bad else 0)
"""

_FRESH_WORDS = """
import importlib
import sys
from datetime import datetime
from types import SimpleNamespace as held
for name in sys.argv[2].split(","):
    try:
        importlib.import_module(name)
    except ImportError:
        pass
from core.gui import record_words
record_words.record_label("simulation", "", None, prior_label="p")
record_words.record_label("calibration", "", datetime(2026, 10, 3, 14, 2))
record_words.record_label("prior", "p", None)
made = held(created="2026-10-03T21:02:11+00:00")
record_words.label_of(held(kind="fdt", name="", manifest=made, body={"study": "sweep"}))
record_words.label_of(held(kind="prior", name="", manifest=held(created="not a moment")))
roots = sys.argv[1].split(",")
bad = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots))
sys.exit(repr(bad) if bad else 0)
"""


def _imports(module: str) -> set:
    """Every module an import statement names, in ``module`` and in every module of this repository it reaches
    that way, wherever each statement stands; a call to ``__import__`` or ``import_module`` with a written name
    counts as a statement."""
    named, seen, todo = set(), set(), [module]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        path = REPO / (current.replace(".", "/") + ".py")
        package = current.rpartition(".")[0].split(".")
        if not path.is_file():
            path, package = REPO / current.replace(".", "/") / "__init__.py", current.split(".")
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            found = set()
            if isinstance(node, ast.Import):
                found = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                source = ".".join(base + ([node.module] if node.module else []))
                found = {source} | {f"{source}.{alias.name}" for alias in node.names}
            elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and ast.unparse(node.func).rpartition(".")[2] in ("__import__", "import_module")):
                found = {node.args[0].value}
            named |= found
            for name in found:
                parts = name.split(".")
                if parts[0] == "core":
                    todo += [".".join(parts[:n]) for n in range(1, len(parts) + 1)]
    return named


def _reaching(names, forbidden) -> list:
    return sorted(name for name in names if any(name == root or name.startswith(root + ".") for root in forbidden))


def test_the_rules_and_tables_import_no_toolkit_and_no_engine():
    """The places, the keys and the steps' rules load neither the toolkit nor the engine, and the record words
    load no engine (they write a date as the look does). Every import statement of each module is read, wherever
    it stands, and so is every statement of each module of this repository it imports, since a call can miss an
    import inside a branch it does not take. In a pytest process both are loaded already, so the proof is a
    fresh interpreter for each, which imports every module those statements name, from this repository or not,
    and calls the rules down every branch."""
    for module, forbidden in (("core.gui.keys", _RULES_FORBIDDEN), ("core.gui.chain", _RULES_FORBIDDEN),
                              ("core.gui.record_words", _WORDS_FORBIDDEN)):
        found = _reaching(_imports(module), forbidden)
        assert not found, f"{module} imports {found}, itself or through a module of this repository it imports"
    rules = _imports("core.gui.keys") | _imports("core.gui.chain")
    for code, named, forbidden in ((_FRESH_RULES, rules, _RULES_FORBIDDEN),
                                   (_FRESH_WORDS, _imports("core.gui.record_words"), _WORDS_FORBIDDEN)):
        r = subprocess.run([sys.executable, "-c", code, ",".join(forbidden), ",".join(sorted(named))],
                           cwd=str(REPO), capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr


# the places and the keys
def test_the_parts_and_steps_are_the_rails_in_order():
    """Nine parts and six steps, in the rail's order, under the names the window shows, each with its keys,
    and each part but Home with the one line Home's table gives it; and the two pages of Effective temperature."""
    assert [p.key for p in keys.PARTS] == ["home", "inference", "temperature", "live", "reduction", "records",
                                           "models", "settings", "help"]
    assert [p.name for p in keys.PARTS] == ["Home", "Parameter inference", "Effective temperature",
                                            "Live simulation", "Reduction map", "Records", "Your own models",
                                            "Settings", "Help"]
    assert [p.keys[0] for p in keys.PARTS] == [f"Ctrl+{n}" for n in range(1, 10)]
    assert keys.PARTS[8].keys == ("Ctrl+9", "F1") and all(len(p.keys) == 1 for p in keys.PARTS[:8])
    assert [p.line for p in keys.PARTS] == [
        "",
        "Infer a cell's model parameters from recordings, in six steps.",
        "Measure it on one cell, or in a sweep study over two parameters; compare saved runs.",
        "Watch a cell's bundle move as it is simulated; save a video.",
        "The analytical map from the Nadrowski model to the Hopf normal form.",
        "Everything PRISM has written, and what each record was made from.",
        "Define a model of your own.",
        "The appearance.",
        "What each part's settings mean, and the keys.",
    ]
    assert [s.key for s in keys.STEPS] == ["model", "prior", "training", "checks", "infer", "narrowing"]
    assert [s.name for s in keys.STEPS] == ["Model and options", "Prior", "Training", "Checks", "Inference",
                                            "Narrowing round"]
    assert [s.keys for s in keys.STEPS] == [(f"Alt+{n}",) for n in range(1, 7)]
    assert all(s.line == "" for s in keys.STEPS)
    assert keys.PAGES == {"temperature": (keys.Place("one_cell", "One cell"), keys.Place("sweep", "Sweep study"))}
    assert keys.part("records") is keys.PARTS[5] and keys.step("infer") is keys.STEPS[4]
    assert [keys.step_number(s.key) for s in keys.STEPS] == [1, 2, 3, 4, 5, 6]
    for lookup, name in ((keys.part, "training"), (keys.step, "records"), (keys.step_number, "home")):
        with pytest.raises(KeyError):
            lookup(name)


def test_the_window_shortcuts_are_the_places_keys_and_three_more():
    """One shortcut for each key of a part or a step, then the two keys between the rail and the page and the
    key to the run, no key twice, each a key sequence the toolkit reads back as written; and the rows Help
    prints, in their order and words."""
    from PySide6.QtGui import QKeySequence
    sequences = [sequence for sequence, _ in keys.BINDINGS]
    assert len(set(sequences)) == len(sequences), "a key is bound twice"
    portable = QKeySequence.SequenceFormat.PortableText
    assert [QKeySequence.fromString(s, portable).toString(portable) for s in sequences] == sequences
    assert keys.BINDINGS == (
        ("Ctrl+1", "part home"), ("Ctrl+2", "part inference"), ("Ctrl+3", "part temperature"),
        ("Ctrl+4", "part live"), ("Ctrl+5", "part reduction"), ("Ctrl+6", "part records"),
        ("Ctrl+7", "part models"), ("Ctrl+8", "part settings"), ("Ctrl+9", "part help"), ("F1", "part help"),
        ("Alt+1", "step model"), ("Alt+2", "step prior"), ("Alt+3", "step training"), ("Alt+4", "step checks"),
        ("Alt+5", "step infer"), ("Alt+6", "step narrowing"),
        ("F6", "rail or page"), ("Shift+F6", "rail or page"), ("Ctrl+R", "run"),
    )
    assert keys.KEYS == (
        ("F6, Shift+F6", "Moves the keyboard between the rail and the page."),
        ("Ctrl+1 to Ctrl+9", "Opens the nine parts in rail order."),
        ("Alt+1 to Alt+6", "Opens steps 1 to 6 from anywhere."),
        ("Ctrl+R", "Goes to the run, whenever the foot is a button."),
        ("F1", "Opens Help."),
        ("Up, Down, Home, End",
         "In the rail, moves through the parts and, when shown, the steps, in the order seen."),
        ("Enter, Space", "In the rail, opens the row."),
        ("Escape", "In a question, the safe answer; closes an open list; in the rail, returns the keyboard to the "
                   "page; otherwise nothing. It never moves between pages and never stops a run."),
    )


def test_a_place_is_written_as_the_window_names_it():
    """A part by its name, a step by its number and name, a page of a part by both names, and several places
    joined so that each keeps its own comma."""
    assert keys.place_words("training") == "step 3, Training"
    assert keys.place_words("records") == "Records"
    assert keys.place_words("inference") == "Parameter inference"
    assert keys.place_words("temperature/sweep") == "Effective temperature, Sweep study"
    assert keys.place_words("temperature/one_cell") == "Effective temperature, One cell"
    assert keys.place_words("inference/narrowing") == "step 6, Narrowing round"
    assert keys.places_words(("training", "narrowing")) == "step 3, Training, or step 6, Narrowing round"
    assert keys.places_words(("help",)) == "Help"
    assert keys.places_words(("model", "infer", "live")) == (
        "step 1, Model and options, or step 5, Inference, or Live simulation")
    for unknown in ("training/sweep", "temperature/compare", "records/prior", "nowhere", "inference/home"):
        with pytest.raises(KeyError):
            keys.place_words(unknown)


# the six steps
def _session(**held):
    fields = dict(draft=None, cfg=None, inf_prior=None, posterior=None, observation=None)
    fields.update(held)
    return SimpleNamespace(**fields)


def _record(name, *, truncation=None):
    """A held record by its name; a posterior's network carries the region it was narrowed to, or None."""
    return SimpleNamespace(name=name, posterior=SimpleNamespace(truncation=truncation))


_DRAFT = SimpleNamespace(model="NADROWSKI")
_PRIOR, _POSTERIOR, _OBSERVATION = _record("retrain_prior"), _record("retrain"), _record("cell_7")
_NARROWED = _record("retrain_round", truncation=object())
_EMPTY = {}
_MODEL = {"draft": _DRAFT}
_WITH_PRIOR = {"draft": _DRAFT, "inf_prior": _PRIOR}
_TRAINED = {"draft": _DRAFT, "inf_prior": _PRIOR, "posterior": _POSTERIOR}
_OBSERVED = {"draft": _DRAFT, "inf_prior": _PRIOR, "posterior": _POSTERIOR, "observation": _OBSERVATION}
_ROUND = {"draft": _DRAFT, "inf_prior": _PRIOR, "posterior": _NARROWED, "observation": _OBSERVATION}

# (session, the rule's options, step, (marker, word, detail, rail words, can start, done), the spoken sentence)
_RULE_TABLE = {
    "any step, its run live": (
        _WITH_PRIOR, {"live": "training"}, "training",
        ("now", "Now", "Running", "Now: running", True, False), "Step 3, Training. Now: running"),
    "any step, its run stopping": (
        _WITH_PRIOR, {"live": "training", "stopping": True}, "training",
        ("now", "Now", "Stopping", "Now: stopping", True, False), "Step 3, Training. Now: stopping"),
    "any step, a live run before a change not applied": (
        _WITH_PRIOR, {"live": "prior", "pending": frozenset({"prior"})}, "prior",
        ("now", "Now", "Running", "Now: running", True, False), "Step 2, Prior. Now: running"),
    "any step, a change not applied": (
        _MODEL, {"pending": frozenset({"model"})}, "model",
        ("warn", "Attention", "Changed, not applied", "Changed, not applied", True, False),
        "Step 1, Model and options. Attention: changed, not applied"),
    "any step, a live run elsewhere changes nothing": (
        _WITH_PRIOR, {"live": "temperature"}, "training",
        ("now", "Now", "Setting up", "Now: setting up", True, False), "Step 3, Training. Now: setting up"),
    "any step, its run live before its needs are met": (
        _MODEL, {"live": "training"}, "training",
        ("now", "Now", "Running", "Now: running", False, False), "Step 3, Training. Now: running"),
    "any step, a change not applied before its needs are met": (
        _EMPTY, {"pending": frozenset({"prior"})}, "prior",
        ("warn", "Attention", "Changed, not applied", "Changed, not applied", False, False),
        "Step 2, Prior. Attention: changed, not applied"),
    "step 1, a built-in model": (
        _MODEL, {}, "model",
        ("done", "Done", "Nadrowski", "Done: Nadrowski", True, True), "Step 1, Model and options. Done: Nadrowski"),
    "step 1, a model of your own": (
        {"draft": SimpleNamespace(model="SHM")}, {}, "model",
        ("done", "Done", "SHM", "Done: SHM", True, True), "Step 1, Model and options. Done: SHM"),
    "step 1, in hand": (
        _EMPTY, {}, "model",
        ("now", "Now", "Choose a model", "Now: choose a model", True, False),
        "Step 1, Model and options. Now: choose a model"),
    "step 2, a prior": (
        _WITH_PRIOR, {}, "prior",
        ("done", "Done", "retrain_prior", "Done: retrain_prior", True, True), "Step 2, Prior. Done: retrain_prior"),
    "step 2, a prior with no name": (
        {"draft": _DRAFT, "inf_prior": SimpleNamespace(name="")}, {}, "prior",
        ("done", "Done", "unnamed", "Done: unnamed", True, True), "Step 2, Prior. Done: unnamed"),
    "step 2, a prior named by the label given": (
        _WITH_PRIOR, {"label": lambda record: f"the {record.name} record"}, "prior",
        ("done", "Done", "the retrain_prior record", "Done: the retrain_prior record", True, True),
        "Step 2, Prior. Done: the retrain_prior record"),
    "step 2, no model": (
        _EMPTY, {}, "prior",
        ("later", "To come", "Needs a model", "Needs a model", False, False), "Step 2, Prior. To come: needs a model"),
    "step 2, in hand": (
        _MODEL, {}, "prior",
        ("now", "Now", "Choose a prior", "Now: choose a prior", True, False), "Step 2, Prior. Now: choose a prior"),
    "step 3, a posterior": (
        _TRAINED, {}, "training",
        ("done", "Done", "retrain", "Done: retrain", True, True), "Step 3, Training. Done: retrain"),
    "step 3, a narrowed posterior": (
        _ROUND, {}, "training",
        ("done", "Done", "retrain_round, narrowed", "Done: retrain_round, narrowed", True, True),
        "Step 3, Training. Done: retrain_round, narrowed"),
    "step 3, a posterior before a training that ended early": (
        _TRAINED, {"ended": 4}, "training",
        ("done", "Done", "retrain", "Done: retrain", True, True), "Step 3, Training. Done: retrain"),
    "step 3, a posterior with no prior in use": (
        {"draft": _DRAFT, "posterior": _POSTERIOR}, {}, "training",
        ("done", "Done", "retrain", "Done: retrain", False, True), "Step 3, Training. Done: retrain"),
    "step 3, no prior": (
        _MODEL, {}, "training",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False),
        "Step 3, Training. To come: needs a prior"),
    "step 3, a training that ended early": (
        _WITH_PRIOR, {"ended": 4}, "training",
        ("half", "Unfinished", "Simulations saved to batch 4", "Unfinished: batch 4", True, False),
        "Step 3, Training. Unfinished: simulations saved to batch 4"),
    "step 3, a training that ended early, its batch a large number": (
        _WITH_PRIOR, {"ended": 9500}, "training",
        ("half", "Unfinished", "Simulations saved to batch 9,500", "Unfinished: batch 9,500", True, False),
        "Step 3, Training. Unfinished: simulations saved to batch 9,500"),
    "step 3, in hand": (
        _WITH_PRIOR, {}, "training",
        ("now", "Now", "Setting up", "Now: setting up", True, False), "Step 3, Training. Now: setting up"),
    "step 3, open but not in hand": (
        {"inf_prior": _PRIOR}, {}, "training",
        ("later", "To come", "", "To come", True, False), "Step 3, Training. To come"),
    "step 4, no prior": (
        _MODEL, {"checks": object()}, "checks",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False), "Step 4, Checks. To come: needs a prior"),
    "step 4, a posterior and no prior in use": (
        {"draft": _DRAFT, "posterior": _POSTERIOR}, {}, "checks",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False), "Step 4, Checks. To come: needs a prior"),
    "step 4, a prior and no posterior": (
        _WITH_PRIOR, {"checks": object()}, "checks",
        ("later", "To come", "Needs a posterior", "Needs a posterior", False, False),
        "Step 4, Checks. To come: needs a posterior"),
    "step 4, never in hand": (
        _TRAINED, {}, "checks",
        ("later", "To come", "", "To come", True, False), "Step 4, Checks. To come"),
    "step 5, an observation": (
        _OBSERVED, {}, "infer",
        ("done", "Done", "cell_7", "Done: cell_7", True, True), "Step 5, Inference. Done: cell_7"),
    "step 5, no posterior": (
        _WITH_PRIOR, {}, "infer",
        ("later", "To come", "Needs a posterior", "Needs a posterior", False, False),
        "Step 5, Inference. To come: needs a posterior"),
    "step 5, in hand": (
        _TRAINED, {}, "infer",
        ("now", "Now", "Setting up", "Now: setting up", True, False), "Step 5, Inference. Now: setting up"),
    "step 6, narrowed": (
        _ROUND, {"observation_on_offer": True}, "narrowing",
        ("done", "Done", "retrain_round", "Done: retrain_round", True, True),
        "Step 6, Narrowing round. Done: retrain_round"),
    "step 6, nothing held": (
        _EMPTY, {"observation_on_offer": True}, "narrowing",
        ("later", "To come", "Needs a posterior", "Needs a posterior", False, False),
        "Step 6, Narrowing round. To come: needs a posterior"),
    "step 6, no posterior": (
        _WITH_PRIOR, {"observation_on_offer": True}, "narrowing",
        ("later", "To come", "Needs a posterior", "Needs a posterior", False, False),
        "Step 6, Narrowing round. To come: needs a posterior"),
    "step 6, no prior": (
        {"draft": _DRAFT, "posterior": _POSTERIOR}, {}, "narrowing",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False),
        "Step 6, Narrowing round. To come: needs a prior"),
    "step 6, no observation on offer": (
        _OBSERVED, {}, "narrowing",
        ("later", "To come", "Needs an observation", "Needs an observation", False, False),
        "Step 6, Narrowing round. To come: needs an observation"),
    "step 6, open and never in hand": (
        _OBSERVED, {"observation_on_offer": True}, "narrowing",
        ("later", "To come", "", "To come", True, False), "Step 6, Narrowing round. To come"),
}


@pytest.mark.parametrize("held, options, step, expected, spoken", list(_RULE_TABLE.values()), ids=list(_RULE_TABLE))
def test_each_step_reads_as_the_rule_table_says(held, options, step, expected, spoken):
    """One case for each rule of a step's state, the first rule that holds deciding; Checks reads "Needs a
    prior" with no prior in use, then "Needs a posterior" with no posterior, and "To come" otherwise, with no
    count, while the checks are not counted."""
    states = chain.step_states(_session(**held), **options)
    assert [s.key for s in states] == [s.key for s in keys.STEPS] and [s.number for s in states] == [1, 2, 3, 4, 5, 6]
    state = states[keys.step_number(step) - 1]
    assert (state.marker, state.word, state.detail, state.rail_words, state.can_start, state.done) == expected
    assert state.spoken == spoken


def test_the_rule_never_raises_on_what_a_session_may_hold():
    """Every session field a bare object, every field None, every field missing, no session at all, each with
    and without the record words naming what is held: six states come back, as a tuple of states that cannot
    change, each with a marker the look draws and the word that marker is read with, and Home finds its next
    step. A name that is not text reads as unnamed, and a record's body or moment that is not what a record
    holds is passed over."""
    from core.gui import paint
    words = {"done": "Done", "now": "Now", "later": "To come", "warn": "Attention", "err": "Error",
             "half": "Unfinished"}
    fields = ("draft", "cfg", "inf_prior", "posterior", "observation")
    sessions = [_session(**{name: object() for name in fields}), _session(), SimpleNamespace(), None]
    sessions += [_session(**{name: object()}) for name in fields]
    for session in sessions:
        for label in (None, record_words.label_of):
            states = chain.step_states(session, label=label, observation_on_offer=True)
            assert [s.key for s in states] == [s.key for s in keys.STEPS]
            for state in states:
                assert state.marker in paint.MARKERS and words[state.marker] == state.word, state
                assert state.spoken.startswith(f"Step {state.number}, ")
            assert isinstance(chain.next_step(session, states), chain.Next)
            assert chain.first_time(has_records=False, has_runs=False, session=session) in (True, False)
    every = chain.step_states(sessions[0])
    assert isinstance(every, tuple) and hash(every) == hash(chain.step_states(sessions[0])), (
        "the states are not a tuple of states that cannot change")
    assert [s.rail_words for s in every] == ["Done: unnamed", "Done: unnamed", "Done: unnamed", "To come",
                                             "Done: unnamed", "Needs an observation"]
    odd = _session(draft=SimpleNamespace(model=7), inf_prior=SimpleNamespace(name=7),
                   posterior=SimpleNamespace(name=["retrain"], posterior=None), observation=SimpleNamespace(name=b"x"))
    states = chain.step_states(odd, observation_on_offer=True)
    assert [states[n].detail for n in (0, 1, 2, 4)] == ["unnamed"] * 4, "a name that is not text was kept"
    made = SimpleNamespace(created=12345)
    assert record_words.label_of(SimpleNamespace(kind="fdt", name="", manifest=made, body=["sweep"])) == (
        "Unnamed effective-temperature measurement")


def test_the_step_in_hand_is_the_first_open_step_and_never_the_narrowing_round():
    """The first step neither done nor unable to start: never Checks while the checks are not counted, never
    the narrowing round, which is optional; and the step in hand is the one step marked Now when nothing runs."""
    open_steps = [chain.StepState(s.key, n, "done", "Done", "x", "Done: x", True, True)
                  for n, s in enumerate(keys.STEPS, 1)]
    open_steps[3] = chain.StepState("checks", 4, "now", "Now", "", "Now", True, False)
    open_steps[5] = chain.StepState("narrowing", 6, "now", "Now", "", "Now", True, False)
    assert chain.step_in_hand(open_steps) is None, "Checks or the narrowing round was taken for the step in hand"
    assert chain.step_in_hand(()) is None
    for held, in_hand in ((_EMPTY, "model"), (_MODEL, "prior"), (_WITH_PRIOR, "training"), (_TRAINED, "infer"),
                          (_OBSERVED, None), (_ROUND, None), ({"inf_prior": _PRIOR}, "model")):
        states = chain.step_states(_session(**held), observation_on_offer=True)
        assert chain.step_in_hand(states) == in_hand, held
        assert [s.key for s in states if s.marker == "now"] == ([in_hand] if in_hand else []), held
    states = chain.step_states(_session(**_WITH_PRIOR), live="training")
    assert chain.step_in_hand(states) == "training"
    states = chain.step_states(_session(**_WITH_PRIOR), ended=4)
    assert chain.step_in_hand(states) == "training" and states[2].marker == "half"
    states = chain.step_states(_session(**_MODEL), pending=frozenset({"model"}))
    assert [s.marker for s in states[:2]] == ["warn", "later"] and chain.step_in_hand(states) == "model", (
        "a step whose change was not applied was passed over for the step in hand")
    blocked = (chain.StepState("model", 1, "later", "To come", "Needs a model", "Needs a model", False, False),
               chain.StepState("prior", 2, "later", "To come", "", "To come", True, False))
    assert chain.step_in_hand(blocked) == "prior", "a step that cannot start was taken for the step in hand"


def test_the_built_in_models_are_written_as_the_window_names_them():
    """The three built-in models by their names on screen, a model of your own as it was stored, and the
    window's table of names holds exactly the engine's built-in models (its list gains your own once loaded)."""
    from core import config, registry
    assert chain.model_label("NADROWSKI") == "Nadrowski"
    assert chain.model_label("HOPF") == "Hopf"
    assert chain.model_label("BP") == "BP"
    assert chain.model_label("SHM") == "SHM"
    assert sorted(chain._MODEL_NAMES) == sorted(m for m in config.VALID_MODELS if not registry.is_user_model(m))
    assert chain.model_label("") == chain.model_label(None) == chain.model_label(object()) == "unnamed"


# Home
_LIVE = ("and no other run can start until it ends. You can still move around and read results. Keep PRISM open; "
         "you can minimise it.")


def test_home_picks_its_one_next_step_in_order():
    """A live run first, wherever it runs; then the steps in their order: no model, no prior, no posterior, no
    observation; otherwise nothing is waiting, with the narrowing round offered, or Records once narrowed."""
    cases = [
        (_EMPTY, {}, ("inference", "training"),
         ("Next: nothing until the run ends", f"A run is going in step 3, Training, {_LIVE}", "Go to the run",
          "inference", "training")),
        (_OBSERVED, {}, ("temperature", "sweep"),
         ("Next: nothing until the run ends", f"A run is going in Effective temperature, Sweep study, {_LIVE}",
          "Go to the run", "temperature", "sweep")),
        (_EMPTY, {}, ("live", None),
         ("Next: nothing until the run ends", f"A run is going in Live simulation, {_LIVE}", "Go to the run",
          "live", None)),
        (_EMPTY, {}, None,
         ("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
          "Open Model and options", "inference", "model")),
        (_MODEL, {}, None, ("Next: build or load a prior", "", "Open Prior", "inference", "prior")),
        (_WITH_PRIOR, {}, None,
         ("Next: start a training", "The Training page says what it will cost before it starts.", "Set up training",
          "inference", "training")),
        (_WITH_PRIOR, {"ended": 4}, None,
         ("Next: start a training", "The Training page says what it will cost before it starts.", "Set up training",
          "inference", "training")),
        (_TRAINED, {}, None,
         ("Next: run the posterior on a cell or on your recordings", "", "Open Inference", "inference", "infer")),
        (_OBSERVED, {"observation_on_offer": True}, None,
         ("Next: nothing is waiting",
          "A narrowing round is optional: it trains a second posterior that is valid only near one observation.",
          "Open Narrowing round", "inference", "narrowing")),
        (_ROUND, {"observation_on_offer": True}, None,
         ("Next: nothing is waiting", "A narrowing round was made: the posterior in use is the narrowed one.",
          "Open Records", "records", None)),
    ]
    for held, options, live_place, expected in cases:
        session = _session(**held)
        found = chain.next_step(session, chain.step_states(session, **options), live_place=live_place)
        assert (found.heading, found.sentence, found.button, found.part, found.page) == expected, (held, live_place)


def test_the_first_time_holds_only_with_nothing_on_disk_and_nothing_loaded():
    """No record, no run in the history and no model chosen: only then is it the first time."""
    assert chain.first_time(has_records=False, has_runs=False, session=_session()) is True
    assert chain.first_time(has_records=False, has_runs=False, session=None) is True
    assert chain.first_time(has_records=True, has_runs=False, session=_session()) is False
    assert chain.first_time(has_records=False, has_runs=True, session=_session()) is False
    assert chain.first_time(has_records=False, has_runs=False, session=_session(draft=_DRAFT)) is False
    with pytest.raises(TypeError):
        chain.first_time(False, False, _session())


# what a record is called
def test_a_record_is_called_by_its_name_or_by_its_kind_and_date(tmp_path):
    """A name when it has one; with none, its kind and when it was made, or for a simulation set the prior it
    was simulated for. A record the session holds is labelled from its manifest, its moment turned from the
    store's time into local time, and the chain names it so. A record made today still carries its year."""
    from core.artifacts import ArtifactStore
    at = datetime(2026, 10, 3, 14, 2, 11)
    assert record_words.record_label("calibration", "", at) == "Unnamed calibration, 3 Oct 2026, 14:02"
    assert record_words.record_label("calibration", "cal_7", at) == "cal_7"
    assert record_words.record_label("inference", "", None) == "Unnamed inference result"
    assert record_words.record_label("fdt", "", at, study="sweep") == "Unnamed sweep, 3 Oct 2026, 14:02"
    assert record_words.record_label("fdt", "", at) == (
        "Unnamed effective-temperature measurement, 3 Oct 2026, 14:02")
    assert record_words.record_label("simulation", "", at, prior_label="retrain_prior") == (
        "Simulation set for retrain_prior")
    assert record_words.record_label("simulation", "", at) == "Simulation set for a prior that is no longer here"
    assert record_words.record_label("simulation", "set_a", at, prior_label="retrain_prior") == "set_a"
    local = at.astimezone()                                      # 14:02 here, whatever the zone of this computer
    store = ArtifactStore(tmp_path, clock=lambda: local.astimezone(timezone.utc))
    with store.create("calibration", None, name="") as w:
        w.body = {"results": {}}
    held = store.load_calibration(w.id)
    assert record_words.label_of(held) == "Unnamed calibration, 3 Oct 2026, 14:02"
    with store.create("fdt", None, name="") as w:
        w.body = {"study": "sweep", "settings": {}, "seed": 7, "grid": None, "points": None, "offgrid": None,
                  "notices": [], "compared": None, "complete": True, "results": {}}
    assert record_words.label_of(store.load_fdt(w.id)) == "Unnamed sweep, 3 Oct 2026, 14:02"
    assert record_words.label_of(SimpleNamespace(kind="prior", name="retrain_prior", manifest=None)) == (
        "retrain_prior")
    assert record_words.label_of(object()) == "Unnamed record"
    states = chain.step_states(_session(draft=_DRAFT, inf_prior=held), label=record_words.label_of)
    assert states[1].rail_words == "Done: Unnamed calibration, 3 Oct 2026, 14:02"
    assert states[1].spoken == "Step 2, Prior. Done: Unnamed calibration, 3 Oct 2026, 14:02"
    # a record made today still carries its year, never its weekday
    recent = datetime.now().replace(second=0, microsecond=0)
    said = record_words.record_label("calibration", "", recent)
    assert said.startswith(f"Unnamed calibration, {recent.day} ") and said.endswith(
        f" {recent.year}, {recent:%H:%M}"), said
    # a record made five hours from this computer's zone is written in this computer's time, whatever its zone
    here = datetime(2026, 10, 3, 14, 2).astimezone().utcoffset()
    away = timezone(here + timedelta(hours=5) if here <= timedelta(hours=8) else here - timedelta(hours=5))
    moment = datetime(2026, 10, 3, 14, 2, 11, tzinfo=away)
    made = SimpleNamespace(kind="calibration", name="", manifest=SimpleNamespace(created=moment.isoformat()))
    assert record_words.label_of(made) == f"Unnamed calibration, 3 Oct 2026, {moment.astimezone():%H:%M}"


def test_each_store_kind_has_one_word():
    """Every kind the store keeps has its one word and no two kinds share one; an effective-temperature record
    takes the word of its study."""
    from core.artifacts.store import KIND_DIRS
    assert set(record_words.KIND_WORDS) == set(KIND_DIRS)
    assert len(set(record_words.KIND_WORDS.values())) == len(record_words.KIND_WORDS)
    assert record_words.KIND_WORDS == {
        "prior": "prior", "simulation": "simulation set", "posterior": "posterior", "observation": "observation",
        "calibration": "calibration", "inference": "inference result", "diagnostic": "check",
        "fdt": "effective-temperature measurement"}
    assert [record_words.kind_word("fdt", study) for study in ("single", "sweep", "comparison", None)] == [
        "effective-temperature measurement", "sweep", "comparison", "effective-temperature measurement"]
    assert record_words.kind_word("prior", "sweep") == "prior"
    assert record_words.kind_word("crossval") == "record"


# the main button of each step's page, by the step
_MAIN = {"model": "btn_config", "prior": "btn_prior", "training": "btn_post", "checks": "btn_validate",
         "infer": "btn_infer", "narrowing": "btn_round"}


def _pressable(panel) -> list:
    """The buttons of a step's page that a person can press now: shown, not greyed, and not a help mark, which
    only shows its words."""
    from PySide6.QtWidgets import QAbstractButton, QPushButton, QToolButton
    return [button for button in panel.findChildren(QAbstractButton)
            if isinstance(button, (QPushButton, QToolButton)) and button.isVisibleTo(panel) and button.isEnabled()
            and button.objectName() != "helpBadge"]


def test_every_step_opens_on_any_session_and_no_press_on_it_starts_a_run(tiny_run, monkeypatch):
    """No step is disabled. On four sessions -- an empty one; a model chosen, by the first step's own button; a
    configuration and a prior; and a posterior with them, the tiny run's -- each of the six steps opens as the
    page shown and the rail marks it, and its main button is greyed exactly when the step cannot start. On a step
    that cannot start, each button that can be pressed is pressed, with every question answered by the guard and
    every file chooser cancelled, and the session built again before each press: no run is started, nothing is
    raised, no drawing error is caught and nothing is shown but a question or a notice."""
    from core.gui import reported
    from core.gui.panels.base_panel import BasePanel
    from core.gui.session import SbiSession
    from tests._fixtures import all_asked, answer_chooser, build_window, forget_asked, qt_app
    qt_app()
    started, raised = [], []
    monkeypatch.setattr(BasePanel, "dispatch", lambda panel, *args, **kwargs: started.append(type(panel).__name__))
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: raised.append(repr(error)))
    answer_chooser(monkeypatch, "")
    window = build_window()
    inference = window.inference_screen
    first = inference.config_panel
    first.model_combo.setCurrentText("SBITEST")
    window.go_to("inference", "model")
    first.btn_config.click()
    draft = inference.session.draft
    assert getattr(draft, "model", None) == "SBITEST", "the first step's button chose no model"
    cfg = tiny_run.cfg.copy_for_run()
    cfg.clear_ground_truth()

    def hold(session):
        """The session ``session`` names, built again: 0 empty, 1 a model, 2 with a prior, 3 with a posterior."""
        if session == 0:
            inference.session = SbiSession()
            inference.refresh_gates()
            return
        inference.new_draft(draft)
        if session >= 2:
            inference.install_config(cfg)
            inference.session.inf_prior = tiny_run.prior
            if session == 3:
                inference.session.posterior = tiny_run.posterior
            inference.refresh_gates()

    pressed = []
    for session in range(4):
        for step, panel in zip(keys.STEPS, inference.panels()):
            hold(session)
            window.go_to("inference", step.key)
            state = inference.step_states()[keys.step_number(step.key) - 1]
            assert (window.current_place(), inference.pages.currentWidget()) == (("inference", step.key), panel), (
                session, step.key)
            assert window.shell.rail.row(step.key).property("ring_left"), (session, step.key)
            assert getattr(panel, _MAIN[step.key]).isEnabled() == state.can_start, (session, step.key)
            if state.can_start:
                continue
            for number in range(len(_pressable(panel))):
                hold(session)
                window.go_to("inference", step.key)
                button = _pressable(panel)[number]
                forget_asked()
                button.click()
                asked = [(box.kind, box.title) for box in all_asked()]
                assert (started, raised, list(reported.ERRORS)) == ([], [], []), (session, step.key, button.text())
                assert all(kind in ("question", "notice") for kind, _title in asked), (session, step.key, asked)
                pressed.append((session, step.key, button.text()))
    assert pressed, "no button of a step that cannot start was pressed"


def test_refresh_gates_says_the_new_states_once_and_the_rail_and_home_hear_them(monkeypatch):
    """One run of the gates runs every step's own gate first, in the steps' order, and then says the six steps'
    states once: the rules' own for the session, with the narrowing round's observation on offer as its list
    says. The rail's rows read them, and Home, in sight, reads its next step from them at once."""
    from PySide6.QtCore import Qt
    from core.gui.session import SbiSession
    from tests._fixtures import build_window, qt_app
    qt_app()
    window = build_window()
    inference, rail = window.inference_screen, window.shell.rail
    heard = []
    for step, panel in zip(keys.STEPS, inference.panels()):
        monkeypatch.setattr(panel, "refresh_local_gates",
                            lambda kept=panel.refresh_local_gates, key=step.key: (heard.append(key), kept())[1])
    inference.states_changed.connect(heard.append)
    window.setAttribute(Qt.WA_DontShowOnScreen)
    try:
        window.show()
        inference.session = SbiSession(draft=SimpleNamespace(model="NADROWSKI"))
        heard.clear()
        inference.refresh_gates()
        states = chain.step_states(inference.session, label=record_words.label_of,
                                   observation_on_offer=inference.tsnpe_panel.obs_picker.has_entries())
        assert heard == [step.key for step in keys.STEPS] + [states], heard
        assert inference.step_states() == states
        assert [rail.row(step.key).accessibleName() for step in keys.STEPS] == [state.spoken for state in states]
        assert window.home_screen.next_heading.label.text() == "Next: build or load a prior"
    finally:
        window.hide()


def test_parameter_inference_opens_the_step_in_hand_then_the_step_last_shown():
    """Parameter inference opened with no step shows the step in hand at its first opening in a launch and the step
    last shown at every later one, a step that cannot start yet among them; a first opening at a named step counts
    as the first. With no step in hand the first opening shows the first step. The rail marks the step shown."""
    from core.gui.session import SbiSession
    from tests._fixtures import build_window, qt_app
    qt_app()
    draft = SimpleNamespace(model="NADROWSKI")
    made = SimpleNamespace(created="2026-10-03T14:02:00+00:00")

    def held(kind):
        return SimpleNamespace(kind=kind, id=f"{kind[0]}1", name=kind, manifest=made,
                               posterior=SimpleNamespace(truncation=None))

    def window_holding(session):
        window = build_window()
        window.inference_screen.session = session
        window.inference_screen.refresh_gates()
        return window

    def opened(window, *moves):
        for move in moves:
            window.go_to(*move)
        return window.current_place(), window.inference_screen.current_step()

    window = window_holding(SbiSession(draft=draft))
    assert opened(window, ("inference",)) == (("inference", "prior"), "prior"), (
        "the first opening is not at the step in hand")
    assert opened(window, ("inference", "infer"), ("live",), ("inference",)) == (("inference", "infer"), "infer"), (
        "a later opening is not at the step last shown")
    assert window.shell.rail.row("infer").property("ring_left"), "the rail does not mark the step shown"
    named = window_holding(SbiSession(draft=draft))
    assert opened(named, ("inference", "checks"), ("home",), ("inference",)) == (("inference", "checks"), "checks"), (
        "a first opening at a named step did not count as the first")
    done = window_holding(SbiSession(draft=draft, cfg=object(), inf_prior=held("prior"), posterior=held("posterior"),
                                     observation=held("observation")))
    assert chain.step_in_hand(done.inference_screen.step_states()) is None
    assert opened(done, ("inference",)) == (("inference", "model"), "model"), (
        "with no step in hand the first opening is not at the first step")
