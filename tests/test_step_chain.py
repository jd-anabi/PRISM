"""The rules and tables with no widget: the window's places and keys, the six steps' states, Home's one next step
and first-time test, and what a record is called; that the run's model loads no widget; and the six steps as the
built window holds them.

Torch-free at its module level: the modules under test load neither the toolkit (the record words aside) nor the
engine, and a fresh interpreter proves it, since this process already holds both. A session is the window's own
session class, made by tests/_window_helpers.py. A test that reads the store's own tables or writes a record imports
them inside itself, and so does a test that builds the window.
"""
import ast
import itertools
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.gui import chain, keys, record_words
from tests._window_helpers import session as _session

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
from core.gui.run.model import RunStatus
run = "20261008T190211-training"
runs = (RunStatus(state="running", word="Check", place=("inference", "checks")),
        RunStatus(state="stopping", word="Training", name="retrain_3", place=("inference", "training")),
        RunStatus(state="retrying", word="Training", name="retrain_3", place=("inference", "training")),
        RunStatus(state="running", word="Training", place=("inference", "training"), needs_you="Nothing heard."),
        RunStatus(state="ended", word="Training", outcome="finished", run=run, place=("inference", "training")),
        RunStatus(state="ended", word="Training", outcome="stopped", run=run, place=("inference", "training")),
        RunStatus(state="not_seen", word="Training", outcome="not seen to end", run=run, place=("records", None)),
        RunStatus(state="running", word="Run"), RunStatus())
some = held(draft=held(model="NADROWSKI"), cfg=None, inf_prior=None, posterior=None, observation=None)
every = held(draft=held(model="SHM"), cfg=None, inf_prior=held(name="p"), observation=held(name="o"),
             posterior=held(name="q", posterior=held(truncation=1)))
for session, options in ((some, dict(live="prior", kept=held(kind="continue", batch=3, epoch=None),
                                     pending=frozenset({"model"}))), (every, {}), (None, {}),
                         (some, dict(live="training", retrying=True))):
    states = chain.step_states(session, observation_on_offer=True, **options)
    said = [state.spoken for state in states]
    chain.next_step(session, states)
    for status in runs:
        chain.next_step(session, states, run=status)
    chain.first_time(has_records=False, has_runs=False, session=session)
last = held(model="NADROWSKI", chi_mode=True, bounds="sub\\\\master.txt", prior="p", prior_words="the prior p",
            posterior="q", posterior_words="the posterior q", plan="retrain")
for picks in (last, held(model="SHM", chi_mode=None, bounds=None, prior=None, posterior=None, plan=None), None):
    for session in (None, some):
        chain.next_step(session, chain.step_states(session), picks=picks)
    if picks is not None:
        chain.last_time_sentence(picks)
said = [chain.model_label("HOPF"), keys.places_words(("training", "narrowing")), keys.place_words("inference/infer")]
early = held(draft=held(model=""), cfg=None, inf_prior=held(name="p"), posterior=None, observation=None)
chain.step_states(early, kept=held(kind="resume", epoch=9500, batch=None), label=str)
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
record_words.label_of(held(kind="diagnostic", name="", manifest=made, body={"variant": "laplace"}))
record_words.label_of(held(kind="prior", name="", manifest=held(created="not a moment")))
roots = sys.argv[1].split(",")
bad = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots))
sys.exit(repr(bad) if bad else 0)
"""

_FRESH_RUN = """
import importlib
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as held
for name in sys.argv[2].split(","):
    try:
        importlib.import_module(name)
    except ImportError:
        pass
from core.gui import texts
from core.gui.run import ends, model, questions, readings, retry
moment = [datetime(2026, 10, 8, 19, 2, 11, tzinfo=timezone(timedelta(hours=2)))]
def clock():
    return moment[0]
def estimate(per, size):
    return (0.5, "a batch") if per == "batch" else (30.0, "an epoch")
def row(desc, stats="", total=None, depth=0, solver=False):
    return held(desc=desc, stats=stats, total=total, row=depth, informative=bool(total and total > 1), is_solver=solver)
def epoch(n, since):
    return (f"[fit] epoch {n}: training loss -1.5, validation loss -1.25, took 30.0 s, epochs since the best {since}",
            "info")
stages = (model.Stage("Rotation", ("rotation_begins",)),
          model.Stage("Simulation", ("simulation_writing",), model.SIMULATION_ROW),
          model.Stage("Fitting", ("simulation_complete",), "fit_epoch"),
          model.Stage("Saving", (model.FIT_ENDED, "sbi_converged")))
facts = {"network": (8, 2), "observation": "chi", "memory_limit": 6.0, "cadence": 10}
runs = [model.RunModel(model.RunSpec("Training", ("inference", "training"), name="t", stages=stages, patience=2,
                                     epoch_limit=9, facts=facts).with_call(len, (), {}),
                       clock=clock, estimate=estimate, expected=expected)
        for expected in ((5, "about 5 epochs: given"), None)]
def look(run):
    run.tick()
    said = (run.state, run.stage, run.outcome, run.sentence(), run.stage_line(), run.axis(), run.health(),
            run.messages(), run.needs_you(), run.epochs(), run.best_epoch(), run.batches_done(),
            run.simulation_folder(), run.began_after_first(), run.fit_not_saved(), run.saves_passed(),
            run.stage_kind(), run.now(), run.stop_bound(), run.last_epoch_ends(), run.fit_ended_by(),
            run.expected_epochs(), run.doing())
    status = model.RunStatus(state="running", word=run.spec.word, name=run.spec.name)
    said += (questions.stop_question(status, run), questions.close_question(status, run),
             questions.shutdown_reason(status, run))
    if run.state == "stopping":
        stopping = model.RunStatus(state="stopping", word=run.spec.word, stop_requested=True)
        said += (questions.stopping_line(stopping, run), questions.close_question(stopping, run))
    failed = model.RunStatus(state="ended", word=run.spec.word, name=run.spec.name, outcome="failed",
                             ended=run.now(), continue_from=7, kept_fit=run.kept_fit)
    said += (retry.failure_words(failed, run, RuntimeError("CUDA out of memory"),
                                 is_oom=lambda error: "out of memory" in str(error)),
             retry.failure_words(failed, run, OSError("No space left on device.")), retry.message(ValueError()))
    waiting = model.RunStatus(state="retrying", word=run.spec.word, name=run.spec.name, continue_from=7,
                              kept_fit=run.kept_fit, trying=2, try_at=run.now() + timedelta(minutes=3))
    said += (retry.failure_words(waiting, run, None), questions.close_question(waiting, run),
             retry.failure_words(replace(failed, reopen=True), run, RuntimeError("x")),
             retry.made_in(run.spec, held(cfg=None, inf_prior=None)))
    moment[0] += timedelta(seconds=50)
    return said
tag = "training batch 3/40 [t_scale=1, T=1, n_fine=8, N_points=8, rows=8]"
for run in runs:
    look(run)
    for line in ("[budget] 40 batches x 8 rows = 320 training rows",
                 "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...",
                 "[fisher] eigenvalue spread (best/worst direction): 3",
                 "[checkpoint] writing to /scratch/set every 10 batches (~0.1 GiB total, 9.0 GiB free)"):
        run.feed([(line, "info")])
        look(run)
    for done in range(0, 41, 4):
        run.rows([row(model.SIMULATION_ROW, f"{done}/40 [00:01<00:02]", 40), row("step (batch=8)", "1/9", 9, 1, True)])
        run.solver(done)
        look(run)
    run.stopping(True)
    look(run)
    run.stopping(False)
    run.trying(True)
    look(run)
    run.trying(False)
    run.set_earlier(((1, 80),))
    run.feed([("[patho] batch 2: 1 new pathological trajectorie(s) -- 1 non-finite, 0 exactly constant, 0 over 1e+06 "
               "in magnitude, of 24 simulated", "warning"),
              (f"[mem] {tag}: peak allocated 1.00 GiB, peak reserved 1.50 GiB, 9.00/16.00 GiB reported free "
               "(optimistic on Windows), learned cap none", "info"),
              (f"{tag}: OOM at simulation batch 8; retrying in chunks of 4. Original: out of memory", "warning"),
              (f"UserWarning: {tag}: chi: 3/16 probes masked (below 3 drive cycles, at/above Nyquist, out of band, "
               "or a non-finite lock-in).", "warning"),
              ("library: rejection: Only 0.5% proposal samples were accepted.", "warning"),
              ("[chi] masked probes: 3 of 16 (18.8%) over 1 batches, this process only (no simulation cache); per "
               "batch 18.8-18.8%, median 18.8%", "info"),
              ("[checkpoint] complete: 40 batches in /scratch/set. Safe to delete once the posterior is saved; "
               "keeping it lets you retrain the flow without re-simulating.", "info"),
              ("[fit] saving is switched off: this training is not saved as it goes and cannot be resumed.", "info")])
    look(run)
    moment[0] += timedelta(hours=2)
    look(run)
    for n in range(1, 10):
        run.feed([epoch(n, 0 if n < 8 else n - 7), ("[fit] saved after pass %d." % n, "info")])
        look(run)
    run.rows([row("Training neural network. Epochs trained: 8")])
    look(run)
    run.ended("stopped", kept_fit=held(id="x", passes=8, resumed=False, saved_here=True, processes=1))
    look(run)
cell = model.RunModel(model.RunSpec("Check", ("inference", "checks"), name="c",
                                    stages=(model.Stage("Checks"), model.Stage("Calibrating"))),
                      clock=clock)
for line in ("Writing", "the next line"):
    cell.feed([(line, "info")])
    look(cell)
own = model.RunModel(model.RunSpec("Export", ("inference", "infer")), clock=clock)
own.set_stage("Watching")
look(own)
counted = model.RunModel(model.RunSpec("Check", ("inference", "checks")), clock=clock)
counted.rows([row("Calibrating", "3/9 [00:01<00:02]", 9)])
look(counted)
counted.ended("finished")
look(counted)
for others in ("Prior build", "Check", "Inference", "Run", "Narrowing round"):
    look(model.RunModel(model.RunSpec(others, ("inference", "checks"), name="n"), clock=clock))
look(model.RunModel(model.RunSpec("Prior load", ("inference", "prior"), stop_asks=False), clock=clock))
said = [questions.keep_question(held(passes=n, resumed=resumed, saved_here=here, processes=2))
        for n, resumed, here in ((1, False, True), (62, True, False), (64, True, True))]
said += [readings.sentence(readings.read(text, "warning"), count=2) for text in (
    "[fit] saved after pass 3.",
    f"{tag}: batch FAILED after both halving retries (1/3). Waiting 6s and re-running the whole batch. Original: x")]
said += [texts.duration_text(s) for s in (0, 61, 3700, 200000)] + [texts.number_text(2048.5)]
said += [texts.date_text(clock()), texts.date_text(clock(), clock())]
said += [model.driving_row([row("a", "1/3", 3)]), model.fit_counter([row("b")]), model.moved(1, 2)]
said += [model.IDLE.live, model.RunStatus(state="ended").unacknowledged_end]
oom = lambda error: "out of memory" in str(error)
narrowing = model.RunSpec("Narrowing round", ("inference", "narrowing"), kwargs={"n_directions": 4, "level": 0.9})
said += [retry.is_tried_again(RuntimeError("out of memory"), runs[0].spec, is_oom=oom),
         retry.only_reopening(RuntimeError("out of memory"), kept_before=(1, 0), kept_after=(1, 0), is_oom=oom,
                              tried=retry.TRIES),
         retry.gained((1, 0), (1, 2)), retry.continued_call(runs[0].spec), retry.resumed_call(narrowing, "k"),
         retry.is_device_failure(ValueError()), retry.PAUSES_S]
at = moment[0]
fit = {"epochs": 9, "best_epoch": 7, "ended_by": "limit", "expected": 12, "simulations": 320, "kept_id": "k",
       "passes": 8, "discarded": False}
for outcome, stage, word in (("finished", "Saving", "Training"), ("stopped", "Simulation", "Training"),
                             ("stopped", "Fitting", "Training"), ("stopped", "Saving", "Narrowing round"),
                             ("stopped", None, "Measurement"), ("failed", "Fitting", "Training"),
                             ("failed", "Rotation", "Check"), ("not seen to end", "Simulation", "Training"),
                             ("not seen to end", "Spontaneous", "Measurement")):
    record = {"id": "r", "word": word, "name": "n", "place": ["inference", "training"], "outcome": outcome,
              "stage": stage, "started": at, "ended": at.isoformat(), "last_heard": at, "stop_reason": None,
              "tries": [{"ended": at, "batches_after": 4}], "simulation_set": {"batches": 40, "at_end": 4,
                                                                                 "per_batch": 8},
              "fit": fit, "result": {"kind": "posterior", "id": "q", "name": "", "asked": "n"},
              "failure": {"kind": "other", "message": "x."}}
    said += [ends.words(record, at, too_late=True),
             ends.words({**record, "stop_reason": "Windows shut down", "tries": [], "fit": None}, at)]
    said += [ends.foot(ends.status_of(record, at), at)]
for status in (model.IDLE, model.RunStatus(state="running", word="Training", stage_words="Simulating"),
               model.RunStatus(state="running", word="Prior load", loading=True),
               model.RunStatus(state="retrying", word="Training"),
               model.RunStatus(state="running", word="Training", needs_you="Nothing heard."),
               model.RunStatus(state="stopping", word="Training", stage_words="Stopping."),
               model.RunStatus(state="ended", word="Training", outcome="finished", ended=at)):
    said += [ends.foot(status, at)]
roots = sys.argv[1].split(",")
bad = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in roots))
sys.exit(repr(bad) if bad else 0)
"""

_FRESH_THIS_COMPUTER = """
import importlib
import sys
for name in sys.argv[2].split(","):
    try:
        importlib.import_module(name)
    except ImportError:
        pass
from core.gui import this_computer
words = "a reason " * 40
said = (this_computer.keep_awake(True), this_computer.block_shutdown(None, words), this_computer.flash(None),
        this_computer.state(), this_computer.keep_awake(False), this_computer.block_shutdown(None, None),
        this_computer.state())
assert said == (False, False, False, (True, this_computer.fitted(words)), False, False, (False, None)), said
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
    """The places, the keys, the steps' rules and what a reopened window can offer load neither the toolkit nor the
    engine, and the record words load no engine (they write a date as the look does). Every import statement of each
    module is read, wherever it stands, and so is every statement of each module of this repository it imports, since
    a call can miss an import inside a branch it does not take. In a pytest process both are loaded already, so the
    proof is a fresh interpreter for each, which imports every module those statements name, from this repository or
    not, and calls the rules down every branch."""
    for module, forbidden in (("core.gui.keys", _RULES_FORBIDDEN), ("core.gui.chain", _RULES_FORBIDDEN),
                              ("core.gui.step_facts", _RULES_FORBIDDEN), ("core.gui.continuing", _RULES_FORBIDDEN),
                              ("core.gui.record_words", _WORDS_FORBIDDEN)):
        found = _reaching(_imports(module), forbidden)
        assert not found, f"{module} imports {found}, itself or through a module of this repository it imports"
    rules = (_imports("core.gui.keys") | _imports("core.gui.chain") | _imports("core.gui.step_facts")
             | _imports("core.gui.continuing"))
    for code, named, forbidden in ((_FRESH_RULES, rules, _RULES_FORBIDDEN),
                                   (_FRESH_WORDS, _imports("core.gui.record_words"), _WORDS_FORBIDDEN)):
        r = subprocess.run([sys.executable, "-c", code, ",".join(forbidden), ",".join(sorted(named))],
                           cwd=str(REPO), capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr


def test_the_run_model_and_its_readings_import_no_toolkit_and_no_engine():
    """The run's model, the table of the engine's lines it reads, the words a stop and a close ask, the words and the
    rules of a failure, and the date and number words load neither the toolkit, nor the engine, nor its settings:
    the window's thread reads a run's lines through them while the run holds the engine, and they are tested with no
    window. Every import statement of each module is read, wherever it stands, with every statement of each module
    of this repository it imports; then a fresh interpreter imports every module those statements name, builds a
    training's model and models of runs of other kinds, feeds them their lines, rows, solver counts and clock, and
    reads every read-out at each step, the questions of a stop and a close among them with the reason Windows is
    given against ending the session, the words of the run's failure and of its tries, the command line its page
    sets off, and the question of keeping a training a stop kept; the rules of the tries and of repeating a run's
    call into its page's session; and how each kind of run's end is said, from its history, and the foot of the
    rail in each state of a run."""
    modules = ("core.gui.texts", "core.gui.run.readings", "core.gui.run.model", "core.gui.run.questions",
               "core.gui.run.retry", "core.gui.run.ends")
    for module in modules:
        assert (REPO / (module.replace(".", "/") + ".py")).is_file(), f"{module} is not there to be read"
        found = _reaching(_imports(module), _RULES_FORBIDDEN)
        assert not found, f"{module} imports {found}, itself or through a module of this repository it imports"
    named = set().union(*(_imports(module) for module in modules))
    r = subprocess.run([sys.executable, "-c", _FRESH_RUN, ",".join(_RULES_FORBIDDEN), ",".join(sorted(named))],
                       cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_requests_made_of_windows_load_no_torch():
    """The one module that makes PRISM's requests of Windows for a run loads no torch: the window's thread makes them
    while the run holds the engine. Every import statement it holds is read, with every statement of each module of
    this repository it imports; then a fresh interpreter imports every module those statements name and makes each
    request, with no window, which off the toolkit's real platform reaches nothing of Windows and answers False."""
    assert (REPO / "core" / "gui" / "this_computer.py").is_file(), "core.gui.this_computer is not there to be read"
    found = _reaching(_imports("core.gui.this_computer"), ("torch",))
    assert not found, f"core.gui.this_computer imports {found}, itself or through a module of this repository"
    r = subprocess.run([sys.executable, "-c", _FRESH_THIS_COMPUTER, "torch",
                        ",".join(sorted(_imports("core.gui.this_computer")))],
                       cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# the places and the keys
def test_the_parts_and_steps_are_the_rails_in_order():
    """Six parts and six steps, in the rail's order, under the names the window shows, each with its keys, the
    parts' keys numbered in the rail's order, and each part but Home with the one line Home's table gives it; and no
    part with pages of its own."""
    assert [p.key for p in keys.PARTS] == ["home", "inference", "records", "models", "settings", "help"]
    assert [p.name for p in keys.PARTS] == ["Home", "Parameter inference", "Records", "Your own models", "Settings",
                                            "Help"]
    assert [p.keys[0] for p in keys.PARTS] == [f"Ctrl+{n}" for n in range(1, 7)]
    assert keys.PARTS[5].keys == ("Ctrl+6", "F1") and all(len(p.keys) == 1 for p in keys.PARTS[:5])
    assert [p.line for p in keys.PARTS] == [
        "",
        "Infer a cell's model parameters from recordings, in six steps.",
        "Everything PRISM has written, and what each record was made from.",
        "Define a model of your own.",
        "Appearance, and this computer.",
        "What each part's settings mean, and the keys.",
    ]
    assert [s.key for s in keys.STEPS] == ["model", "prior", "training", "checks", "infer", "narrowing"]
    assert [s.name for s in keys.STEPS] == ["Model and options", "Prior", "Training", "Checks", "Inference",
                                            "Narrowing round"]
    assert [s.keys for s in keys.STEPS] == [(f"Alt+{n}",) for n in range(1, 7)]
    assert all(s.line == "" for s in keys.STEPS)
    assert keys.part("records") is keys.PARTS[2] and keys.step("infer") is keys.STEPS[4]
    assert [keys.step_number(s.key) for s in keys.STEPS] == [1, 2, 3, 4, 5, 6]
    for lookup, name in ((keys.part, "training"), (keys.step, "records"), (keys.step_number, "home")):
        with pytest.raises(KeyError):
            lookup(name)


def test_the_window_shortcuts_are_the_places_keys_and_three_more():
    """One shortcut for each key of a part or a step, then the two keys between the rail and the page and the
    key to the run, no key twice, each a key sequence the toolkit reads back as written, and none for Ctrl+7 to
    Ctrl+9, past the six parts; and the rows Help prints, in their order and words."""
    from PySide6.QtGui import QKeySequence
    sequences = [sequence for sequence, _ in keys.BINDINGS]
    assert len(set(sequences)) == len(sequences), "a key is bound twice"
    portable = QKeySequence.SequenceFormat.PortableText
    assert [QKeySequence.fromString(s, portable).toString(portable) for s in sequences] == sequences
    assert keys.BINDINGS == (
        ("Ctrl+1", "part home"), ("Ctrl+2", "part inference"), ("Ctrl+3", "part records"),
        ("Ctrl+4", "part models"), ("Ctrl+5", "part settings"), ("Ctrl+6", "part help"), ("F1", "part help"),
        ("Alt+1", "step model"), ("Alt+2", "step prior"), ("Alt+3", "step training"), ("Alt+4", "step checks"),
        ("Alt+5", "step infer"), ("Alt+6", "step narrowing"),
        ("F6", "rail or page"), ("Shift+F6", "rail or page"), ("Ctrl+R", "run"),
    )
    assert not {"Ctrl+7", "Ctrl+8", "Ctrl+9"} & set(sequences), "a key past the six parts is bound"
    assert keys.KEYS == (
        ("F6, Shift+F6", "Moves the keyboard between the rail and the page."),
        ("Ctrl+1 to Ctrl+6", "Opens the six parts in rail order."),
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
    """A part by its name, a step by its number and name, a step named as a page of Parameter inference by the step
    alone, and several places joined as a sentence names them after its "in": each keeps its own comma, a step after
    another place begins with its own number, and a part after another place takes its own "in". A page of a part
    that has none, the effective temperature's pages among them, is no place."""
    assert keys.place_words("training") == "step 3, Training"
    assert keys.place_words("records") == "Records"
    assert keys.place_words("inference") == "Parameter inference"
    assert keys.place_words("inference/narrowing") == "step 6, Narrowing round"
    assert keys.places_words(("training", "narrowing")) == "step 3, Training, or step 6, Narrowing round"
    assert keys.places_words(("help",)) == "Help"
    assert keys.places_words(("model", "infer", "records")) == (
        "step 1, Model and options, or step 5, Inference, or in Records")
    assert keys.places_words(("records", "inference/narrowing")) == "Records, or step 6, Narrowing round"
    for unknown in ("training/sweep", "temperature/sweep", "temperature/one_cell", "temperature", "live", "reduction",
                    "records/prior", "nowhere", "inference/home"):
        with pytest.raises(KeyError):
            keys.place_words(unknown)


# the six steps
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
        _WITH_PRIOR, {"live": "records"}, "training",
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
        _TRAINED, {"kept": SimpleNamespace(kind="continue", batch=4)}, "training",
        ("done", "Done", "retrain", "Done: retrain", True, True), "Step 3, Training. Done: retrain"),
    "step 3, a posterior with no prior in use": (
        {"draft": _DRAFT, "posterior": _POSTERIOR}, {}, "training",
        ("done", "Done", "retrain", "Done: retrain", False, True), "Step 3, Training. Done: retrain"),
    "step 3, no prior": (
        _MODEL, {}, "training",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False),
        "Step 3, Training. To come: needs a prior"),
    "step 3, a training that ended early": (
        _WITH_PRIOR, {"kept": SimpleNamespace(kind="continue", batch=4)}, "training",
        ("half", "Unfinished", "Simulations saved to batch 4", "Unfinished: batch 4", True, False),
        "Step 3, Training. Unfinished: simulations saved to batch 4"),
    "step 3, a training that ended early, its batch a large number": (
        _WITH_PRIOR, {"kept": SimpleNamespace(kind="stopped", batch=9500)}, "training",
        ("half", "Unfinished", "Simulations saved to batch 9,500", "Unfinished: batch 9,500", True, False),
        "Step 3, Training. Unfinished: simulations saved to batch 9,500"),
    "step 3, in hand": (
        _WITH_PRIOR, {}, "training",
        ("now", "Now", "Setting up", "Now: setting up", True, False), "Step 3, Training. Now: setting up"),
    "step 3, open but not in hand": (
        {"inf_prior": _PRIOR}, {}, "training",
        ("later", "To come", "", "To come", True, False), "Step 3, Training. To come"),
    "step 4, no prior, no summary": (
        _MODEL, {}, "checks",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False), "Step 4, Checks. To come: needs a prior"),
    "step 4, a posterior and no prior in use, no summary": (
        {"draft": _DRAFT, "posterior": _POSTERIOR}, {}, "checks",
        ("later", "To come", "Needs a prior", "Needs a prior", False, False), "Step 4, Checks. To come: needs a prior"),
    "step 4, a prior and no posterior, no summary": (
        _WITH_PRIOR, {}, "checks",
        ("later", "To come", "Needs a posterior", "Needs a posterior", False, False),
        "Step 4, Checks. To come: needs a posterior"),
    "step 4, no summary, never in hand": (
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
    """One case for each rule of a step's state, the first rule that holds deciding; handed no summary of the checks,
    Checks reads "Needs a prior" with no prior in use, then "Needs a posterior" with no posterior, and "To come"
    otherwise, with no count, as before the checks were counted."""
    states = chain.step_states(_session(**held), **options)
    assert [s.key for s in states] == [s.key for s in keys.STEPS] and [s.number for s in states] == [1, 2, 3, 4, 5, 6]
    state = states[keys.step_number(step) - 1]
    assert (state.marker, state.word, state.detail, state.rail_words, state.can_start, state.done) == expected
    assert state.spoken == spoken


def _summary(passed=0, of=6, *, failed=False, look_closer=False, open_=True, can_check=True, second="",
             headline=None, sentence=""):
    """The checks' summary, as the screen hands it to the chain."""
    from core.gui.check_lineage import ChecksSummary
    return ChecksSummary(passed, of, failed, look_closer, open_, can_check, second,
                         headline or f"{passed} of {of} deciding checks passed", sentence)


def test_checks_no_run_can_pass_read_attention_and_leave_the_step_in_hand_to_inference():
    """With no deciding check open, failed or to look closer at, and some that no run can pass for the posterior in use,
    Checks reads "Attention" with its count, the warning's marker, and is never the step in hand: Inference is, and
    Home offers it. With none that no run can pass it reads Done; a check failed or to look closer at, or one still
    open, holds the step in hand as before, the checks no run can pass among them or not."""
    from dataclasses import replace
    trained = _session(**_TRAINED)
    left = replace(_summary(4, open_=False), cannot_pass=2)
    states = chain.step_states(trained, checks=left)
    assert (states[3].marker, states[3].rail_words, states[3].done, states[3].may_be_in_hand) == (
        "warn", "Attention: 4 of 6 passed", False, False)
    assert chain.step_in_hand(states) == "infer"
    assert chain.next_step(trained, states, checks=left).button == "Open Inference"
    states = chain.step_states(trained, checks=replace(_summary(6, open_=False), cannot_pass=0))
    assert (states[3].marker, states[3].rail_words, chain.step_in_hand(states)) == ("done", "Done: 6 of 6 passed",
                                                                                     "infer")
    for summary, marker in ((replace(_summary(3, failed=True, open_=False), cannot_pass=2), "err"),
                            (replace(_summary(3, look_closer=True, open_=False), cannot_pass=2), "warn"),
                            (replace(_summary(3), cannot_pass=2), "now")):
        states = chain.step_states(trained, checks=summary)
        assert (states[3].marker, chain.step_in_hand(states)) == (marker, "checks"), summary


_CHECKS_TABLE = {
    "nothing can be checked yet": (
        _TRAINED, _summary(can_check=False),
        ("later", "To come", "Needs a prior", "Needs a prior", False, False), "Step 4, Checks. To come: needs a prior"),
    "a check failed": (
        _TRAINED, _summary(2, failed=True, look_closer=True, second="1 failed"),
        ("err", "Error", "2 of 6 deciding checks passed (1 failed)", "Error: 2 of 6 passed", True, False),
        "Step 4, Checks. Error: 2 of 6 deciding checks passed (1 failed)"),
    "a check to look closer at": (
        _TRAINED, _summary(2, look_closer=True, second="1 to look closer at"),
        ("warn", "Attention", "2 of 6 deciding checks passed (1 to look closer at)", "Attention: 2 of 6 passed", True,
         False), "Step 4, Checks. Attention: 2 of 6 deciding checks passed (1 to look closer at)"),
    "every deciding check passed": (
        _TRAINED, _summary(6, open_=False),
        ("done", "Done", "6 of 6 deciding checks passed", "Done: 6 of 6 passed", True, True),
        "Step 4, Checks. Done: 6 of 6 deciding checks passed"),
    "every one of the three that apply passed": (
        _TRAINED, _summary(3, 3, open_=False, headline="3 of 3 deciding checks passed; 3 do not apply to this posterior"),
        ("done", "Done", "3 of 3 deciding checks passed; 3 do not apply to this posterior", "Done: 3 of 3 passed", True,
         True), "Step 4, Checks. Done: 3 of 3 deciding checks passed; 3 do not apply to this posterior"),
    "open and in hand": (
        _TRAINED, _summary(2, second="4 not run yet"),
        ("now", "Now", "2 of 6 deciding checks passed (4 not run yet)", "Now: 2 of 6 passed", True, False),
        "Step 4, Checks. Now: 2 of 6 deciding checks passed (4 not run yet)"),
    "open and not in hand": (
        _WITH_PRIOR, _summary(0, second="4 wait for step 3"),
        ("later", "To come", "0 of 6 deciding checks passed (4 wait for step 3)", "To come: 0 of 6 passed", True, False),
        "Step 4, Checks. To come: 0 of 6 deciding checks passed (4 wait for step 3)"),
}


@pytest.mark.parametrize("held, checks, expected, spoken", list(_CHECKS_TABLE.values()), ids=list(_CHECKS_TABLE))
def test_the_checks_step_reads_its_counts(held, checks, expected, spoken):
    """Handed the checks' summary, Checks needs only a prior, and reads in the rail how many deciding checks passed of
    how many: "Error" when one failed, "Attention" when one needs a closer look, "Done" when every one that applies
    passed, else "Now" in hand or "To come"; its detail, which Home's table, its tool tip and its accessible name carry
    whole, is the headline with the summary's second line in brackets. It never reads "Needs a posterior"."""
    states = chain.step_states(_session(**held), checks=checks)
    state = states[keys.step_number("checks") - 1]
    assert (state.marker, state.word, state.detail, state.rail_words, state.can_start, state.done) == expected
    assert state.spoken == spoken and state.may_be_in_hand
    assert "Needs a posterior" not in (state.detail, state.rail_words)


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
    odd_checks = SimpleNamespace(can_check=True, passed=None, of=[], failed="yes", look_closer=1, open=None,
                                 headline=7, second_line=object())
    for session, checks in itertools.product(sessions, (None, object(), odd_checks)):
        for label in (None, record_words.label_of):
            states = chain.step_states(session, label=label, observation_on_offer=True, checks=checks)
            assert [s.key for s in states] == [s.key for s in keys.STEPS]
            for state in states:
                assert state.marker in paint.MARKERS and words[state.marker] == state.word, state
                assert state.spoken.startswith(f"Step {state.number}, ")
            assert isinstance(chain.next_step(session, states, checks=checks), chain.Next)
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
    assert record_words.label_of(SimpleNamespace(kind="diagnostic", name="", manifest=made, body=["laplace"])) == (
        "Unnamed check")


def test_checks_is_in_hand_while_a_deciding_check_that_applies_is_open():
    """The first step neither done nor unable to start: Checks while its checks are counted and one that applies is
    open, and never while they are not counted; never the narrowing round, which is optional, never a step that holds
    something with a choice changed and not applied; and the step in hand is the one step marked Now when nothing
    runs."""
    open_steps = [chain.StepState(s.key, n, "done", "Done", "x", "Done: x", True, True)
                  for n, s in enumerate(keys.STEPS, 1)]
    open_steps[5] = chain.StepState("narrowing", 6, "now", "Now", "", "Now", True, False)
    assert chain.step_in_hand(open_steps) is None, "the narrowing round was taken for the step in hand"
    open_steps[3] = chain.StepState("checks", 4, "now", "Now", "", "Now", True, False)
    assert chain.step_in_hand(open_steps) == "checks", "Checks counted and open was not the step in hand"
    open_steps[3] = chain.StepState("checks", 4, "later", "To come", "", "To come", True, False, False, False)
    assert chain.step_in_hand(open_steps) is None, "Checks not counted was taken for the step in hand"
    assert chain.step_in_hand(()) is None
    trained = _session(**_TRAINED)
    states = chain.step_states(trained, checks=_summary(2))
    assert ([s.marker for s in states[3:5]], chain.step_in_hand(states)) == (["now", "later"], "checks")
    assert (states[4].rail_words, states[4].word) == ("To come", "To come")
    states = chain.step_states(trained, checks=_summary(6, open_=False))
    assert ([s.marker for s in states[3:5]], chain.step_in_hand(states)) == (["done", "now"], "infer")
    states = chain.step_states(trained, checks=_summary(2, failed=True))
    assert (states[3].marker, chain.step_in_hand(states)) == ("err", "checks"), "a check that failed is still open"
    states = chain.step_states(trained)
    assert (states[3].may_be_in_hand, chain.step_in_hand(states)) == (False, "infer")
    states = chain.step_states(_session(**_WITH_PRIOR), checks=_summary(0))
    assert chain.step_in_hand(states) == "training" and states[3].rail_words == "To come: 0 of 6 passed"
    for held, in_hand in ((_EMPTY, "model"), (_MODEL, "prior"), (_WITH_PRIOR, "training"), (_TRAINED, "infer"),
                          (_OBSERVED, None), (_ROUND, None), ({"inf_prior": _PRIOR}, "model")):
        states = chain.step_states(_session(**held), observation_on_offer=True)
        assert chain.step_in_hand(states) == in_hand, held
        assert [s.key for s in states if s.marker == "now"] == ([in_hand] if in_hand else []), held
    for held, in_hand in ((_TRAINED, "checks"), (_OBSERVED, "checks"), (_ROUND, "checks")):
        states = chain.step_states(_session(**held), observation_on_offer=True, checks=_summary(1))
        assert chain.step_in_hand(states) == in_hand, held
        assert [s.key for s in states if s.marker == "now"] == [in_hand], held
    states = chain.step_states(_session(**_WITH_PRIOR), live="training")
    assert chain.step_in_hand(states) == "training"
    states = chain.step_states(_session(**_WITH_PRIOR), kept=SimpleNamespace(kind="continue", batch=4))
    assert chain.step_in_hand(states) == "training" and states[2].marker == "half"
    states = chain.step_states(_session(**_MODEL), pending=frozenset({"model"}))
    assert states[0].pending and [s.marker for s in states[:2]] == ["warn", "now"], (
        "a step that holds a model with a change not applied was not passed over")
    assert chain.step_in_hand(states) == "prior"
    states = chain.step_states(_session(), pending=frozenset({"model"}))
    assert not states[0].pending and chain.step_in_hand(states) == "model", (
        "a step that holds nothing was passed over for a change not applied")
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
# What was in use last time, read by its attributes as Home reads the settings file's picks, and the sentence Home says.
_LAST = SimpleNamespace(model="NADROWSKI", chi_mode=True, bounds="master_tier1.txt", prior="20261001T222900",
                        prior_words="the prior retrain_prior", posterior="20261003T031200",
                        posterior_words="the posterior retrain", plan="retrain")
_LAST_SAID = ("Last time: the model Nadrowski, the frequency response as the kind of observation, the bounds file "
              "master_tier1.txt, the prior retrain_prior, the posterior retrain, the plan retrain. Nothing is loaded "
              "until you press the button.")


def test_home_picks_its_one_next_step_in_order():
    """A live run first, wherever it runs, named by its word and its name; a run with no place leads nowhere and
    is passed over, as nothing running is; then, with no model chosen, what was in use last time, when a model was;
    then the steps in their order: no model, no prior, no posterior, no observation; otherwise nothing is waiting, with
    the narrowing round offered, or Records once narrowed. Only what was in use last time does something in place of
    opening a place."""
    from core.gui.run.model import IDLE, RunStatus
    training = RunStatus(state="running", word="Training", name="retrain_3", place=("inference", "training"))
    cases = [
        (_EMPTY, {}, training,
         ("Next: nothing until the training ends", f"Training retrain_3 is running, {_LIVE}", "Go to the run",
          "inference", "training", None)),
        (_EMPTY, {"picks": _LAST}, training,
         ("Next: nothing until the training ends", f"Training retrain_3 is running, {_LIVE}", "Go to the run",
          "inference", "training", None)),
        (_EMPTY, {"picks": _LAST}, IDLE,
         ("Next: continue with what was in use last time", _LAST_SAID, "Continue with these", "home", None, None)),
        (_EMPTY, {"picks": _LAST}, RunStatus(state="running", word="Run"),
         ("Next: continue with what was in use last time", _LAST_SAID, "Continue with these", "home", None, None)),
        (_MODEL, {"picks": _LAST}, None, ("Next: build or load a prior", "", "Open Prior", "inference", "prior", None)),
        (_OBSERVED, {}, RunStatus(state="running", word="Check", place=("inference", "checks")),
         ("Next: nothing until the check ends", f"Check is running, {_LIVE}", "Go to the run",
          "inference", "checks", None)),
        (_EMPTY, {}, RunStatus(state="starting", word="Prior build", place=("inference", "prior")),
         ("Next: nothing until the prior build ends", f"Prior build is running, {_LIVE}", "Go to the run",
          "inference", "prior", None)),
        (_EMPTY, {}, RunStatus(state="running", word="Run"),
         ("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
          "Open Model and options", "inference", "model", None)),
        (_EMPTY, {}, IDLE,
         ("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
          "Open Model and options", "inference", "model", None)),
        (_MODEL, {}, None, ("Next: build or load a prior", "", "Open Prior", "inference", "prior", None)),
        (_WITH_PRIOR, {}, None,
         ("Next: start a training", "The Training page says what it will cost before it starts.", "Set up training",
          "inference", "training", None)),
        (_WITH_PRIOR, {"kept": SimpleNamespace(kind="continue", batch=4)}, None,
         ("Next: start a training", "The Training page says what it will cost before it starts.", "Set up training",
          "inference", "training", None)),
        (_TRAINED, {}, None,
         ("Next: run the posterior on a cell or on your recordings", "", "Open Inference", "inference", "infer",
          None)),
        (_OBSERVED, {"observation_on_offer": True}, None,
         ("Next: nothing is waiting",
          "A narrowing round is optional: it trains a second posterior that is valid only near one observation.",
          "Open Narrowing round", "inference", "narrowing", None)),
        (_ROUND, {"observation_on_offer": True}, None,
         ("Next: nothing is waiting", "A narrowing round was made: the posterior in use is the narrowed one.",
          "Open Records", "records", None, None)),
    ]
    for held, options, run, expected in cases:
        session, options = _session(**held), dict(options)
        picks = options.pop("picks", None)
        found = chain.next_step(session, chain.step_states(session, **options), run=run, picks=picks)
        assert (found.heading, found.sentence, found.button, found.part, found.page, found.run) == expected, (
            held, run)
        assert found.action == ("continue_with_these" if found.button == "Continue with these" else None), (held, run)


def _offer(kind, **held):
    """An offer of a reopened window as the chain reads it: by its attributes, as ``continuing.Offer`` holds them."""
    from core.gui.continuing import Offer
    return Offer(kind, held.pop("run", None), held.pop("kept", None), held.pop("batch", None),
                 held.pop("sentence", f"what the {kind} says"), held.pop("why_not", ""), **held)


def test_a_resumable_kept_training_makes_step_three_unfinished_with_no_prior():
    """A kept training that can be resumed turns step 3 to "Unfinished" after the epoch it holds, before any prior is
    in use, since a Resume puts its own in use; a training to continue or stopped says the batch its set holds; a set to
    fit again and a kept narrowing round change no step; and step 3 holding a posterior says that first."""
    for kind, held, want in (("resume", dict(epoch=41), ("Training kept after epoch 41", "Unfinished: epoch 41")),
                             ("continue", dict(batch=236), ("Simulations saved to batch 236", "Unfinished: batch 236")),
                             ("stopped", dict(batch=9500), ("Simulations saved to batch 9,500",
                                                            "Unfinished: batch 9,500"))):
        for session in (_session(), _session(**_MODEL)):
            state = chain.step_states(session, kept=_offer(kind, **held))[2]
            assert (state.marker, state.detail, state.rail_words, state.can_start, state.done) == (
                "half", *want, True, False), (kind, state)
    for kind in ("fit_again", "round"):
        state = chain.step_states(_session(**_MODEL), kept=_offer(kind, batch=10_000, epoch=3))[2]
        assert (state.marker, state.detail) == ("later", "Needs a prior"), kind
    state = chain.step_states(_session(**_TRAINED), kept=_offer("resume", epoch=41))[2]
    assert (state.marker, state.detail) == ("done", "retrain")


def test_a_kept_round_makes_step_six_unfinished_before_its_needs():
    """A kept narrowing round that can be resumed turns step 6 to "Unfinished" after the epoch it holds, and a round
    to continue after the batch its set holds, before what the step needs (a posterior, a prior, an observation), since
    the press puts its own in use; it can start; step 3 is not touched by it, and step 6 holding a narrowed posterior
    says that first."""
    from types import SimpleNamespace
    for kept, want in ((_offer("round", kept="k", epoch=41, step="narrowing"),
                        ("Narrowing round kept after epoch 41", "Unfinished: epoch 41")),
                       (_offer("continue", run="r", batch=236, step="narrowing"),
                        ("Simulations saved to batch 236", "Unfinished: batch 236"))):
        for session in (_session(), _session(**_MODEL)):
            states = chain.step_states(session, kept_round=kept)
            state = states[5]
            assert (state.marker, state.detail, state.rail_words, state.can_start, state.done) == (
                "half", *want, True, False), (kept.kind, state)
            assert states[2].marker != "half", "a kept round made step 3 unfinished"
    for kind, held in (("round", dict(epoch=41)), ("continue", dict(batch=236))):
        state = chain.step_states(_session(**_MODEL), kept=_offer(kind, step="narrowing", **held))[2]
        assert state.marker != "half", kind
    narrowed = SimpleNamespace(name="retrain_round1", posterior=SimpleNamespace(truncation=object()))
    state = chain.step_states(_session(**{**_TRAINED, "posterior": narrowed}),
                              kept_round=_offer("round", kept="k", epoch=41, step="narrowing"))[5]
    assert state.marker == "done", state


def test_home_picks_a_kept_round_in_its_order():
    """A kept round is one of the offers Home takes by their age: the newest offer that can be taken stands first,
    a kept round's like any other, above what was in use last time and the step in hand; one that cannot be resumed is
    passed over."""
    last = SimpleNamespace(model="NADROWSKI", chi_mode=False, bounds=None, prior=None, posterior=None, plan=None,
                           prior_words="", posterior_words="")
    session = _session()
    states = chain.step_states(session)
    kept_round = _offer("round", kept="k", epoch=41, label="retrain_round1", step="narrowing")
    training = _offer("continue", run="r", batch=236)
    assert chain.next_step(session, states, picks=last, offers=(kept_round, training)).action == "open_round"
    assert chain.next_step(session, states, picks=last, offers=(training, kept_round)).action == "continue_run"
    refused = _offer("round", kept="k", epoch=41, step="narrowing", why_not="its code has changed")
    assert chain.next_step(session, states, picks=last, offers=(refused,)).action == "continue_with_these"


def test_an_offers_action_is_read_from_its_kind_and_its_step():
    """What a button that takes an offer does is read from the offer's kind and the step whose page takes it, never
    from its kind alone: a kept training resumes, a kept round opens its page, a round's history continues the round,
    and a training's continues the training; anything else no button takes."""
    for kind, step, want in (("resume", "training", "resume"), ("continue", "training", "continue_run"),
                             ("fit_again", "training", "continue_run"), ("round", "narrowing", "open_round"),
                             ("continue", "narrowing", "continue_round"), ("fit_again", "narrowing", "continue_round"),
                             ("stopped", "training", None), ("stopped", "narrowing", None),
                             ("round", "training", None), ("resume", "narrowing", None)):
        assert chain.action_of(_offer(kind, step=step)) == want, (kind, step)


def test_home_names_the_newest_offer_above_what_was_in_use_last_time():
    """The newest offer that can be taken stands above "Continue with these" and the step in hand: a training to
    continue, a set to fit again, a kept training to resume by its name, a kept narrowing round by its name, opened on
    the step that lists it with no command, a narrowing round to continue, an unnamed one by Records' label read as part
    of the heading; a training stopped with its set unfinished, and a kept training that cannot be resumed, are passed
    over, so Home stays ordinary after a stop."""
    last = SimpleNamespace(model="NADROWSKI", chi_mode=False, bounds=None, prior=None, posterior=None, plan=None,
                           prior_words="", posterior_words="")
    session = _session()
    states = chain.step_states(session)
    for offer, want in (
            (_offer("continue", run="r", batch=236), ("Next: continue the training", "what the continue says",
                                                      "Continue from batch 236", "inference", "training",
                                                      "continue_run")),
            (_offer("fit_again", run="r", batch=10_000), ("Next: continue the training", "what the fit_again says",
                                                          "Fit again on the saved simulations", "inference",
                                                          "training", "continue_run")),
            (_offer("resume", kept="k", epoch=41, label="retrain_3"), ("Next: resume the training retrain_3",
                                                                       "what the resume says", "Resume", "inference",
                                                                       "training", "resume")),
            (_offer("round", kept="k", epoch=41, label="retrain_round1", step="narrowing"),
             ("Next: resume the narrowing round retrain_round1", "what the round says", "Open Narrowing round",
              "inference", "narrowing", "open_round")),
            (_offer("continue", run="r", batch=236, step="narrowing"),
             ("Next: continue the narrowing round", "what the continue says", "Continue from batch 236", "inference",
              "narrowing", "continue_round"))):
        found = chain.next_step(session, states, picks=last, offers=(offer,))
        assert (found.heading, found.sentence, found.button, found.part, found.page, found.action) == want, offer.kind
        assert found.offer is offer and found.link is None
        assert not hasattr(found, "command"), "Home's next step still sets a command apart"
    unnamed = _offer("resume", kept="k", epoch=41, label="Unnamed unfinished training, 3 Oct 2026, 01:00", named=False)
    assert chain.next_step(session, states, picks=last, offers=(unnamed,)).heading == (
        "Next: resume the unnamed unfinished training, 3 Oct 2026, 01:00"), "an unnamed one's heading is no sentence"
    passed = (_offer("stopped", run="r", batch=236), _offer("resume", kept="k", why_not="its set is gone"))
    found = chain.next_step(session, states, picks=last, offers=passed)
    assert (found.button, found.action, found.offer) == ("Continue with these", "continue_with_these", None)
    found = chain.next_step(session, states, offers=passed)
    assert (found.heading, found.offer) == ("Next: choose a model", None)


def test_the_next_step_of_a_failure_offers_the_button_and_a_link():
    """A failure, or a run not seen to end, whose end nobody has seen and whose own run something is offered for: the
    button takes the offer and "See what happened" stands beside it, opening the run's page; an offer of another run
    leaves the row as it was; a stopped run's row and a finished one's take no offer."""
    from core.gui.run.model import RunStatus
    run = "20261003T030211-training"
    for outcome, state in (("failed", "ended"), ("not seen to end", "not_seen")):
        status = RunStatus(state=state, word="Training", name="retrain_3", outcome=outcome, run=run,
                           place=("inference", "training"), sentence="It failed.")
        own = _offer("continue", run=run, batch=236)
        found = chain.next_step(_session(), chain.step_states(_session()), run=status,
                                offers=(_offer("resume", kept="k", run="another"), own))
        assert (found.heading, found.button, found.action, found.link, found.run, found.offer) == (
            "Next: see what happened to the training", "Continue from batch 236", "continue_run", "See what happened",
            run, own), outcome
        found = chain.next_step(_session(), chain.step_states(_session()), run=status,
                                offers=(_offer("continue", run="another", batch=5),))
        assert (found.button, found.link, found.action) == ("See what happened", None, None), outcome
        # a kept training of that run stays its own Resume
        kept = _offer("resume", run=run, kept="k", epoch=7, label="retrain_3")
        found = chain.next_step(_session(), chain.step_states(_session()), run=status, offers=(kept,))
        assert (found.button, found.action, found.link, found.offer) == ("Resume", "resume", "See what happened",
                                                                         kept), outcome
        # a kept narrowing round of a round's run opens its page and resumes nothing, as Home's button for one does
        round_run = "20261003T030211-narrowing-round"
        status = RunStatus(state=state, word="Narrowing round", name="retrain_round1", outcome=outcome, run=round_run,
                           place=("inference", "narrowing"), sentence="It failed.")
        kept = _offer("round", run=round_run, kept="k", epoch=41, label="retrain_round1", step="narrowing")
        found = chain.next_step(_session(), chain.step_states(_session()), run=status, offers=(kept,))
        assert (found.heading, found.button, found.action, found.link, found.page, found.offer) == (
            "Next: see what happened to the narrowing round", "Open Narrowing round", "open_round",
            "See what happened", "narrowing", kept), outcome
    for outcome, button in (("stopped", "Go to the run"), ("finished", "See the result")):
        status = RunStatus(state="ended", word="Training", outcome=outcome, run=run, place=("inference", "training"))
        found = chain.next_step(_session(), chain.step_states(_session()), run=status,
                                offers=(_offer("continue", run=run, batch=236),))
        assert (found.button, found.link) == (button, None), outcome


def test_a_stopped_training_reads_unfinished_and_home_stays_ordinary():
    """A training stopped with its set unfinished: step 3 reads "Unfinished: batch 236", and Home names the step in
    hand, as after any stop."""
    session = _session(**_WITH_PRIOR)
    offers = (_offer("stopped", run="r", batch=236),)
    states = chain.step_states(session, kept=offers[0])
    assert states[2].rail_words == "Unfinished: batch 236"
    found = chain.next_step(session, states, offers=offers)
    assert (found.heading, found.button, found.offer) == ("Next: start a training", "Set up training", None)


def test_the_first_time_holds_only_with_nothing_on_disk_and_nothing_loaded():
    """No record, no run in the history and no model chosen: only then is it the first time."""
    assert chain.first_time(has_records=False, has_runs=False, session=_session()) is True
    assert chain.first_time(has_records=False, has_runs=False, session=None) is True
    assert chain.first_time(has_records=True, has_runs=False, session=_session()) is False
    assert chain.first_time(has_records=False, has_runs=True, session=_session()) is False
    assert chain.first_time(has_records=False, has_runs=False, session=_session(draft=_DRAFT)) is False
    with pytest.raises(TypeError):
        chain.first_time(False, False, _session())


def test_home_offers_what_was_in_use_last_time_only_with_nothing_chosen():
    """With no model chosen, Home's next step is what was in use last time when a model was: one button that does
    something in place of opening a place, and stays on Home. With nothing read, or no model in it, Home says to choose
    a model; with a model chosen, changed on its page or not, the step in hand. The first time, which reads the session
    alone, is as it was."""
    choose = ("Next: choose a model", "Nothing is loaded: PRISM starts every launch with nothing loaded.",
              "Open Model and options", "inference", "model", None, None)
    for picks in (None, SimpleNamespace(model=None), SimpleNamespace(model=""), SimpleNamespace(model=object())):
        found = chain.next_step(_session(), chain.step_states(_session()), picks=picks)
        assert (found.heading, found.sentence, found.button, found.part, found.page, found.run, found.action) == (
            choose), picks
    found = chain.next_step(_session(), chain.step_states(_session()), picks=_LAST)
    assert (found.heading, found.sentence, found.button, found.part, found.page, found.run, found.action) == (
        "Next: continue with what was in use last time", _LAST_SAID, "Continue with these", "home", None, None,
        "continue_with_these")
    for held, heading in ((_MODEL, "Next: build or load a prior"), (_WITH_PRIOR, "Next: start a training"),
                          (_TRAINED, "Next: run the posterior on a cell or on your recordings")):
        session = _session(**held)
        found = chain.next_step(session, chain.step_states(session), picks=_LAST)
        assert (found.heading, found.action) == (heading, None), held
    pending = _session(**_MODEL)
    found = chain.next_step(pending, chain.step_states(pending, pending=frozenset({"model"})), picks=_LAST)
    assert found.action is None, "a model held and changed on its page was taken for nothing chosen"
    assert chain.first_time(has_records=False, has_runs=False, session=_session()) is True


def test_a_live_run_and_an_unseen_end_come_before_continue_with_these():
    """A live run, a load among them, one that needs someone, and an end nobody has seen, finished, stopped, failed or
    not seen to end, each come before what was in use last time; a run with no place leads nowhere, as nothing running
    does."""
    from core.gui.run.model import RunStatus
    run = "20261008T190211-training"
    place = ("inference", "training")
    for status, heading in (
            (RunStatus(state="running", word="Training", name="retrain_3", place=place),
             "Next: nothing until the training ends"),
            (RunStatus(state="running", word="Prior load", place=("inference", "prior")),
             "Next: nothing until the prior load ends"),
            (RunStatus(state="running", word="Training", place=place, needs_you="Nothing heard."),
             "Next: look at the training"),
            (RunStatus(state="ended", word="Training", outcome="finished", run=run, place=place),
             "Next: see the training"),
            (RunStatus(state="ended", word="Training", outcome="stopped", run=run, place=place),
             "Next: see what happened to the training"),
            (RunStatus(state="ended", word="Training", outcome="failed", run=run, place=place),
             "Next: see what happened to the training"),
            (RunStatus(state="not_seen", word="Training", outcome="not seen to end", run=run, place=place),
             "Next: see what happened to the training"),
            (RunStatus(state="running", word="Run"), "Next: continue with what was in use last time")):
        found = chain.next_step(_session(), chain.step_states(_session()), run=status, picks=_LAST)
        assert found.heading == heading, status
        assert found.action == ("continue_with_these" if heading.startswith("Next: continue") else None), status


@pytest.mark.parametrize("picks, said", [
    (dict(), "the model Nadrowski, the kind of observation set by the bounds file"),
    (dict(chi_mode=True), "the model Nadrowski, the frequency response as the kind of observation"),
    (dict(chi_mode=False), "the model Nadrowski, the kind of observation set by the bounds file"),
    (dict(model="SHM"), "the model SHM, the kind of observation set by the bounds file"),
    (dict(model="HOPF", chi_mode=True, bounds="cell.txt"),
     "the model Hopf, the frequency response as the kind of observation, the bounds file cell.txt"),
    (dict(bounds="sub\\master_tier1.txt"),
     "the model Nadrowski, the kind of observation set by the bounds file, the bounds file sub/master_tier1.txt"),
    (dict(bounds="master.txt", prior="p1", prior_words="the prior retrain_prior"),
     "the model Nadrowski, the kind of observation set by the bounds file, the bounds file master.txt, the prior "
     "retrain_prior"),
    (dict(bounds="master.txt", prior="p1", prior_words="the unnamed prior, 1 Oct 2026, 22:29", posterior="q1",
          posterior_words="the unnamed posterior, 3 Oct 2026, 03:12"),
     "the model Nadrowski, the kind of observation set by the bounds file, the bounds file master.txt, the unnamed "
     "prior, 1 Oct 2026, 22:29, the unnamed posterior, 3 Oct 2026, 03:12"),
    (dict(plan="retrain"), "the model Nadrowski, the kind of observation set by the bounds file, the plan retrain"),
    (dict(chi_mode=True, bounds="master_tier1.txt", prior="p1", prior_words="the prior retrain_prior",
          posterior="q1", posterior_words="the posterior retrain", plan="retrain"),
     "the model Nadrowski, the frequency response as the kind of observation, the bounds file master_tier1.txt, the "
     "prior retrain_prior, the posterior retrain, the plan retrain"),
    (dict(prior="p1", prior_words=""), "the model Nadrowski, the kind of observation set by the bounds file"),
    (dict(posterior_words="the posterior retrain"),
     "the model Nadrowski, the kind of observation set by the bounds file"),
], ids=["model alone", "frequency response", "set by the bounds file", "a model of your own", "Hopf's file",
        "a sub-folder", "a prior", "unnamed records", "a plan", "everything", "a prior with no words",
        "words with no posterior"])
def test_the_sentence_names_what_the_press_applies_and_leaves_out_what_is_gone(picks, said):
    """Home's sentence names, by the names the window shows, exactly what one press applies and loads: the model, the
    kind of observation it is applied with, which is the bounds file's when the settings file holds none it can read,
    the bounds file with forward slashes, the prior and the posterior by the words a sentence names a record by, and
    the plan; each left out when it is gone."""
    held = dict(model="NADROWSKI", chi_mode=None, bounds=None, prior=None, prior_words="", posterior=None,
                posterior_words="", plan=None)
    held.update(picks)
    assert chain.last_time_sentence(SimpleNamespace(**held)) == (
        f"Last time: {said}. Nothing is loaded until you press the button.")


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
    assert record_words.record_label("diagnostic", "", at) == "Unnamed check, 3 Oct 2026, 14:02"
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
    with store.create("diagnostic", None, name="") as w:
        w.body = {"diagnostic": "identifiability", "variant": "laplace", "settings": {}, "results": {}}
    assert record_words.label_of(store.load_diagnostic(w.id)) == "Unnamed check, 3 Oct 2026, 14:02"
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
    """Every kind the store keeps has its one word and no two kinds share one; a kind it does not keep is a
    record."""
    from core.artifacts.store import KIND_DIRS
    assert set(record_words.KIND_WORDS) == set(KIND_DIRS)
    assert len(set(record_words.KIND_WORDS.values())) == len(record_words.KIND_WORDS)
    assert record_words.KIND_WORDS == {
        "prior": "prior", "simulation": "simulation set", "posterior": "posterior", "observation": "observation",
        "calibration": "calibration", "inference": "inference result", "diagnostic": "check",
        "fit": "unfinished training"}
    assert record_words.kind_word("fdt") == record_words.kind_word("crossval") == "record"


# the main button of each step's page, by the step: its foot's
_MAIN = {"model": "foot.main_button", "prior": "foot.main_button", "training": "foot.main_button",
         "checks": "foot.main_button", "infer": "foot.main_button", "narrowing": "foot.main_button"}


def _main_button(panel, step):
    """The main button of ``panel``, the page of ``step``."""
    found = panel
    for name in _MAIN[step].split("."):
        found = getattr(found, name)
    return found


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
    page shown, neither it nor the stack of the pages disabled, and the rail marks it; its main button is greyed
    exactly when the step cannot start, but that a rebuilt page may grey it for a reason of its own, which it says
    beside it. On a step
    that cannot start, each button that can be pressed is pressed, with every question answered by the guard and
    every file chooser cancelled, and the session built again before each press. A press may show another view of its
    page, so on a page of two views each view is put back before each press, and the buttons that can be pressed are
    read again for each: every view's own are pressed. No run is started, nothing is raised, no drawing error is caught
    and nothing is shown but a question or a notice."""
    from core.gui import reported
    from core.gui.run.controller import RunController
    from core.gui.session import SbiSession
    from tests._fixtures import all_asked, answer_chooser, build_window, forget_asked, install_config, qt_app
    qt_app()
    started, raised = [], []
    # the controller's own start, which every page reaches whether or not its class goes through its base class's dispatch
    monkeypatch.setattr(RunController, "dispatch",
                        lambda held, *args, **kwargs: started.append(type(kwargs.get("host")).__name__))
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, trace: raised.append(repr(error)))
    answer_chooser(monkeypatch, "")
    window = build_window()
    inference = window.part("inference")
    first = inference.config_panel
    first.model_combo.setCurrentText("SBITEST")
    window.go_to("inference", "model")
    first.foot.main_button.click()
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
            install_config(inference, cfg)
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
            button = _main_button(panel, step.key)
            if hasattr(panel, "foot"):
                assert not button.isEnabled() or state.can_start, (session, step.key)
                assert button.isEnabled() or panel.foot.reason.text(), (session, step.key)
            else:
                assert button.isEnabled() == state.can_start, (session, step.key)
            assert panel.isEnabled() and inference.pages.isEnabled(), (session, step.key)
            if state.can_start:
                continue
            for view in {"prior": ("saved", "build"), "training": ("training", "load")}.get(step.key, (None,)):
                done = set()
                while True:
                    hold(session)
                    window.go_to("inference", step.key)
                    if view is not None:
                        panel.show_view(view)    # a press may have shown the other view: each is put back
                    waiting = [button for button in _pressable(panel) if button not in done]
                    if not waiting:
                        break
                    button = waiting[0]
                    done.add(button)
                    forget_asked()
                    button.click()
                    asked = [(box.kind, box.title) for box in all_asked()]
                    assert (started, raised, list(reported.ERRORS)) == ([], [], []), (session, step.key, view,
                                                                                         button.text())
                    assert all(kind in ("question", "notice") for kind, _title in asked), (session, step.key, asked)
                    pressed.append((session, step.key, button.text()))
    assert pressed, "no button of a step that cannot start was pressed"


def test_refresh_gates_says_the_new_states_once_and_the_rail_and_home_hear_them(monkeypatch):
    """One run of the gates runs every step's own gate first, in the steps' order, and then says the six steps'
    states once: the rules' own for the session, with the narrowing round's observation on offer as its held reading
    says. The rail's rows read them, and Home, in sight, reads its next step from them at once."""
    from PySide6.QtCore import Qt
    from core.gui.session import SbiSession
    from tests._fixtures import build_window, qt_app
    qt_app()
    window = build_window()
    inference, rail = window.part("inference"), window.shell.rail
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
                                   observation_on_offer=inference.tsnpe_panel.observation_on_offer())
        assert heard == [step.key for step in keys.STEPS] + [states], heard
        assert inference.step_states() == states
        assert [rail.row(step.key).accessibleName() for step in keys.STEPS] == [state.spoken for state in states]
        assert window.part("home").next_heading.label.text() == "Next: build or load a prior"
    finally:
        window.hide()


def test_parameter_inference_opens_the_step_in_hand_then_the_step_last_shown():
    """Parameter inference opened with no step shows the step in hand at its first opening in a launch and the step
    last shown at every later one, a step that cannot start yet among them; a first opening at a named step counts
    as the first, and a step that is none, refused, counts as no opening. With no step in hand the first opening
    shows the first step. The rail marks the step shown."""
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
        window.part("inference").session = session
        window.part("inference").refresh_gates()
        return window

    def opened(window, *moves):
        for move in moves:
            window.go_to(*move)
        return window.current_place(), window.part("inference").current_step()

    window = window_holding(SbiSession(draft=draft))
    with pytest.raises(KeyError):
        window.go_to("inference", "nowhere")
    assert opened(window, ("inference",)) == (("inference", "prior"), "prior"), (
        "the first opening is not at the step in hand")
    assert opened(window, ("inference", "infer"), ("records",), ("inference",)) == (("inference", "infer"), "infer"), (
        "a later opening is not at the step last shown")
    assert window.shell.rail.row("infer").property("ring_left"), "the rail does not mark the step shown"
    named = window_holding(SbiSession(draft=draft))
    assert opened(named, ("inference", "checks"), ("home",), ("inference",)) == (("inference", "checks"), "checks"), (
        "a first opening at a named step did not count as the first")
    done = window_holding(SbiSession(draft=draft, cfg=object(), inf_prior=held("prior"), posterior=held("posterior"),
                                     observation=held("observation")))
    assert chain.step_in_hand(done.part("inference").step_states()) is None
    assert opened(done, ("inference",)) == (("inference", "model"), "model"), (
        "with no step in hand the first opening is not at the first step")
    counted = window_holding(SbiSession(draft=draft, cfg=object(), inf_prior=held("prior"),
                                        posterior=held("posterior"), observation=held("observation")))
    counted.part("inference").read_checks()
    assert chain.step_in_hand(counted.part("inference").step_states()) == "checks"
    assert opened(counted, ("inference",)) == (("inference", "checks"), "checks"), (
        "with a deciding check open the first opening is not at Checks")


def test_home_says_check_the_posterior_with_the_headline():
    """With a deciding check that applies still open after a training, Home's next step is to check the posterior, the
    headline its sentence and "Open Checks" its button; a live run, an offer to take and what was in use last time come
    before it, as before every step in hand; once every one that applies has passed, the next step is the inference;
    and with no summary, Checks is never Home's next step."""
    from core.gui.run.model import RunStatus
    trained = _session(**_TRAINED)
    checks = _summary(2, headline="2 of 6 deciding checks passed", sentence="Inference can be run now; 4 deciding "
                                                                            "checks are still open.")
    states = chain.step_states(trained, checks=checks)
    found = chain.next_step(trained, states, checks=checks)
    assert (found.heading, found.sentence, found.button, found.part, found.page, found.action) == (
        "Next: check the posterior", "2 of 6 deciding checks passed", "Open Checks", "inference", "checks", None)
    three = _summary(1, 3, headline="1 of 3 deciding checks passed; 3 do not apply to this posterior")
    assert chain.next_step(trained, chain.step_states(trained, checks=three), checks=three).sentence == (
        "1 of 3 deciding checks passed; 3 do not apply to this posterior")
    live = RunStatus(state="running", word="Check", name="Calibration", place=("inference", "checks"))
    assert chain.next_step(trained, states, run=live, checks=checks).heading == "Next: nothing until the check ends"
    offer = _offer("continue", run="20261008T190211-training", batch=4, label="retrain")
    assert chain.next_step(trained, states, offers=(offer,), checks=checks).heading == "Next: continue the training"
    passed = _summary(6, open_=False)
    assert chain.next_step(trained, chain.step_states(trained, checks=passed), checks=passed).heading == (
        "Next: run the posterior on a cell or on your recordings")
    assert chain.next_step(trained, chain.step_states(trained)).heading == (
        "Next: run the posterior on a cell or on your recordings")
    assert chain.next_step(_session(**_WITH_PRIOR), chain.step_states(_session(**_WITH_PRIOR), checks=checks),
                           checks=checks).heading == "Next: start a training"


@pytest.mark.parametrize("step", ["checks", "infer", "narrowing"])
def test_a_later_steps_change_not_applied_is_said_on_that_step_alone_and_never_in_hand(step):
    """A change not applied on step 4, 5 or 6 reads "Changed, not applied" on that step alone, every other step reading
    as it does with none; a step that holds something with such a change is never the step in hand, and the step in hand
    is then the first other one; Checks, which holds nothing, stays in hand as an earlier step that holds nothing
    does, but only while its checks are counted: handed no summary, it is never in hand, a change not applied or
    none."""
    session = _session(**_ROUND)
    for checks, others in ((_summary(2), "checks"), (_summary(6, open_=False), None)):
        plain = chain.step_states(session, checks=checks, observation_on_offer=True)
        states = chain.step_states(session, checks=checks, observation_on_offer=True, pending=frozenset({step}))
        number = keys.step_number(step) - 1
        assert (states[number].marker, states[number].rail_words) == ("warn", "Changed, not applied")
        assert [state for n, state in enumerate(states) if n != number] == [
            state for n, state in enumerate(plain) if n != number], "another step's state moved"
        assert states[number].pending == (step != "checks")
        in_hand = chain.step_in_hand(states)
        if step == "checks":
            assert in_hand == "checks", "a step holding nothing was passed over for a change not applied"
        else:
            assert in_hand == others, (checks, in_hand)
    if step == "checks":
        unread = chain.step_states(session, observation_on_offer=True, pending=frozenset({step}))
        assert (unread[3].rail_words, unread[3].may_be_in_hand) == ("Changed, not applied", False)
        assert chain.step_in_hand(unread) != "checks", "Checks was in hand with no summary of its checks"


def test_steps_four_to_six_keep_the_chains_words_for_a_model_inference_cannot_take(monkeypatch):
    """With a model of your own on step 1's form that parameter inference cannot take, nothing is applied: step 1 still
    reads "Now: choose a model" in the rail, the reason it cannot be applied stands beside its own button, and steps 4
    to 6 read the chain's own words, before and after the checks are read."""
    from core import registry
    from tests._fixtures import build_window, qt_app
    qt_app()
    monkeypatch.setattr(registry, "is_user_model", lambda name: name == "HOPF")
    monkeypatch.setattr(registry, "is_sbi_user_model", lambda name: False)
    monkeypatch.setattr(registry, "user_model_has_forcing", lambda name: False)
    window = build_window()
    inference = window.part("inference")
    first = inference.config_panel
    first.model_combo.setCurrentText("HOPF")
    first.redraw()
    for read in (False, True):
        if read:
            inference.read_checks()
        inference.refresh_gates()
        rail = window.shell.rail
        assert inference.session.draft is None
        assert [rail.row(key).accessibleName() for key in ("model", "checks", "infer", "narrowing")] == [
            "Step 1, Model and options. Now: choose a model", "Step 4, Checks. To come: needs a prior",
            "Step 5, Inference. To come: needs a posterior", "Step 6, Narrowing round. To come: needs a posterior"], read
        assert not first.foot.main_button.isEnabled()
        assert first.foot.reason.text() == "Cannot start: this model cannot be used.", first.foot.reason.text()
