"""The run's model and the table of the engine's lines it reads, with no window and no thread: a run's lines and
progress rows are fed to it, and the window's clock is handed in, so a run of days is read in a moment.

A progress row is the window's own parse of a frame of the engine's progress bar; an epoch's record is the window's
literal copy of the line the training call writes, which the first test holds to the engine's own function.

The run's controller, which feeds the model on the window's thread and writes the run's files, is tested after
them, through the stand-in run of tests/_runs.py: a page's own dispatch, the real worker, and the window's clock
moved by hand; each test's folder of runs is its own.

The run's page, which takes the place of a page's results while its run shows, is tested last, the same way: what
it shows is held to what the model reads, and it is drawn as the look's suite draws, into a picture at each display
scaling (it is a part of the window, which the builders alone make, so the look's own suite cannot draw it).
"""
import ast
import importlib
import importlib.util
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.gui import texts
from core.gui.run import model, readings
from core.gui.run.model import FIT_ENDED, SIMULATION_ROW, RunModel, RunSpec, Stage

REPO = Path(__file__).resolve().parents[1]

# local time with its zone, as the window's clock gives it
_ZONE = timezone(timedelta(hours=-7))
# a simulation set's folder as the engine names it
_FOLDER = r"C:\Users\J\PRISM\Artifacts\simulations\_unnamed__20261008T190211"
# the settings a training batch was simulated at, as the engine names the batch inside a line
_SETTINGS = "[t_scale=0.8123, T=1.042, n_fine=4096, N_points=2048, rows=2048]"
# a training's stages, as the training's page hands them to its run
_TRAINING = (Stage("Rotation", ("rotation_begins",)),
             Stage("Simulation", ("simulation_writing", "simulation_resuming"), SIMULATION_ROW),
             Stage("Fitting", ("simulation_complete", "patho_total", "chi_total", "chi_none"), "fit_epoch"),
             Stage("Saving", (FIT_ENDED, "sbi_converged")))
# One cell's stages, as its page hands them to its run
_CELL = (Stage("Checks"), Stage("Spontaneous", ("fdt_campaign_spontaneous",)),
         Stage("Driven", ("fdt_campaign_driven",)))
# what the window is told of a training's expected epochs, and where the number comes from
_EXPECTED = (130, "about 130 epochs: the last training of this kind, retrain_aug, ran 130")

# One line of each row, as the engine writes it.
_LINES = {
    "budget": "[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows",
    "rotation_begins": "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...",
    "rotation_done": "[fisher] eigenvalue spread (best/worst direction): 412",
    "rotation_reused": "Reusing the Fisher rotation stored with the training checkpoint (236/10000 batches) — NOT "
                       "recomputing it: the rotation's operating points are not reproducible across processes, so a "
                       "fresh V would put the reused rows in a different coordinate than the targets stored beside "
                       "them.",
    "rotation_parent": "[tsnpe] basis: reusing the PARENT posterior's rotation carried with the region (V 3f9a2c1d); "
                       "the Fisher is NOT recomputed for a truncated round -- a fresh V would put the box on axes it "
                       "was never measured on.",
    "simulation_writing": f"[checkpoint] writing to {_FOLDER} every 50 batches (~10.3 GiB total, 120.5 GiB free)",
    "simulation_resuming": f"[checkpoint] resuming at batch 236/10000 from {_FOLDER} (reusing the stored rotation V)",
    "simulation_complete": f"[checkpoint] complete: 10000 batches in {_FOLDER}. Safe to delete once the posterior is "
                           f"saved; keeping it lets you retrain the flow without re-simulating.",
    "winsor": "[winsor] clipped 1,234 of 1,003,520,000 summary elements (0.000%) to their per-column 0.5%/99.5% "
              "percentiles",
    "fit_epoch": "[fit] epoch 12: training loss -3.41274, validation loss -3.39812, took 3541.2 s, epochs since the "
                 "best 0",
    "sbi_converged": "Neural network successfully converged after 131 epochs.",
    "patho_batch": "[patho] batch 6149: 3 new pathological trajectorie(s) -- 12,000 non-finite, 6,450 exactly "
                   "constant, 0 over 1e+06 in magnitude, of 12,595,200 simulated",
    "patho_total": "[patho] run total: 30,720 pathological of 20,480,000 simulated trajectories (0.1500%) -- 20,000 "
                   "non-finite, 10,720 exactly constant, 0 over 1e+06",
    "chi_batch": f"UserWarning: training batch 6150/10000 {_SETTINGS}: chi: 9120/22528 probes masked (below 3 drive "
                 f"cycles, at/above Nyquist, out of band, or a non-finite lock-in).",
    "chi_total": "[chi] masked probes: 8,212,345 of 20,480,000 (40.1%) over 10000 batches, every committed batch of "
                 "the simulation cache; per batch 31.2-48.9%, median 40.0%",
    "chi_none": "[chi] masked probes: none counted, this process only (no simulation cache)",
    "mem": f"[mem] training batch 1/10000 {_SETTINGS}: peak allocated 10.42 GiB, peak reserved 11.10 GiB, "
           f"14.68/15.92 GiB reported free (optimistic on Windows), learned cap none",
    "mem_unavailable": "[mem] after OOM in the simulator: memory statistics unavailable (CUDA error: out of memory)",
    "oom_split_simulation": f"training batch 6151/10000 {_SETTINGS}: OOM at simulation batch 2048; retrying in chunks "
                            f"of 1024, 3.71 GiB reported free (optimistic on Windows). Original: CUDA out of memory",
    "oom_split_rows": f"training batch 6151/10000 {_SETTINGS}: OOM with 2048 rows OUTSIDE the simulator retry; "
                      f"re-running this batch in halves of 1024. Original: CUDA out of memory",
    "batch_failed": f"training batch 237/10000 {_SETTINGS}: batch FAILED after both halving retries (1/3), 0.41 GiB "
                    f"reported free (optimistic on Windows). Waiting 60s and re-running the whole batch. Original: "
                    f"CUDA out of memory",
    "stopping": "[checkpoint] stopping: saving 37 completed batches (6100 -> 6137) before unwinding…",
    "save_put_off": "[checkpoint] deferring the write at batch 6150: no RNG snapshot. The rows are still held and go "
                    "out at the next boundary.",
    "fit_saving_off": "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed.",
    "fit_not_kept": "[fit] the simulation set is not kept (checkpointing off), so this training is not saved as it "
                    "goes and cannot be resumed.",
    "fit_listing_failed": "[fit] the kept trainings could not be listed to name the ones that match these settings "
                          "(JSONDecodeError: Expecting value: line 1 column 1 (char 0)); this run starts a new one.",
    "fit_matches": "[fit] 2 kept training(s) match these settings and can be resumed: 20261011T121000 (stopped, "
                   "after pass 62), 20261002T090000 (stopped unexpectedly, after pass 3); this run starts a new one.",
    "fit_resume_fallback": "[fit] the newest saved pass of the kept training 20261011T121000 could not be read; it "
                           "resumes from the pass before it, pass 39.",
    "fit_resuming": "[fit] resuming the kept training 20261011T121000 after pass 40 (best validation loss -3.412, 2 "
                    "passes since the best); its simulation set is read back first.",
    "fit_mark_failed": "[fit] could not mark the kept training 20261011T121000 as one that cannot be resumed "
                       "(PermissionError: [WinError 32] The process cannot access the file).",
    "fit_folder_left": r"[fit] could not remove the folder C:\Users\J\PRISM\Artifacts\fits\_unnamed__20261011T121000, "
                       r"which holds no kept training (PermissionError: [WinError 32] The process cannot access the "
                       r"file); the store lists it as a leftover to clear.",
    "fit_saved": "[fit] saved after pass 41.",
    "fit_save_failed": "[fit] the last save failed: OSError: [Errno 28] No space left on device; the kept training "
                       "stands at pass 40.",
    "fit_save_failed_nothing_kept": "[fit] the last save failed: OSError: [Errno 28] No space left on device; "
                                    "nothing is kept yet.",
    "fit_stopped_nothing_kept": "[fit] the training stopped before its fit saved a pass, so no kept training was "
                                "made; the simulation set keeps the batches it committed.",
    "fit_stopped_not_saved_here": "[fit] the training stopped before it saved a pass in this process; it is kept as "
                                  "fit 20261011T121000, as it was after pass 40.",
    "fit_stopped_kept": "[fit] the training stopped after pass 62 and is kept as fit 20261011T121000 (stopped on "
                        "purpose).",
    "fit_stop_write_failed": "[fit] could not write the kept training's record as it stopped (OSError: [Errno 28] No "
                             "space left on device).",
    "fit_finish_mark_failed": "[fit] could not mark the kept training 20261011T121000 finished (OSError: [Errno 28] "
                              "No space left on device).",
    "fit_remove_failed": "[fit] the kept training 20261011T121000 could not be removed after its posterior was "
                         "written ([WinError 32] The process cannot access the file); it can be deleted.",
    "fit_no_summary": "[fit] this sbi's trainer has no per-pass summary to read, so no line is written per epoch.",
    "fit_record_skipped": "[fit] an epoch's record could not be written (KeyError: 'epoch_durations_sec'), so the "
                          "epochs of this training are not all said.",
    "text_file_broken": "The run's text file could not be written (OSError: [Errno 28] No space left on device); the "
                        "full text is still on the page.",
    "fdt_sanity_skipped": "Skipping sanity checks.",
    "fdt_campaign_spontaneous": "Campaign 1: spontaneous fluctuations -> PSD",
    "fdt_campaign_driven": "Campaign 2: forced response -> chi via lock-in",
    "sweep_phase": "--- Phase A (temp sweep): spontaneous PSD + omega_0 detection ---",
    "sweep_point_a": "  [A 3/8] temp=1.5 (s=0)",
    "sweep_point_b": "  [B 3/8] temp=1.5",
}

# The engine's lines that begin as the run's lines do and that the window reads as text, each with why: no row of
# the table reads them, and a warning or an error among them is said from the engine as it stands.
_TEXT_ONLY = {
    ("core.orchestrator", "[checkpoint] the stored rotation's eigenvalues come with it"):
        "what a continued simulation set says of the rotation it carries",
    ("core.orchestrator", "[checkpoint] this checkpoint stores no rotation eigenvalues"):
        "the same, for a set written before the eigenvalues were kept",
    ("core.SBI.pipeline", "[checkpoint] state.pt could not be read"): "a warning said from the engine",
    ("core.SBI.pipeline", "[checkpoint] moved "): "what a continued set did with files it does not read",
    ("core.SBI.pipeline", "[checkpoint] no valid RNG snapshot for batch"): "a warning said from the engine",
    ("core.SBI.pipeline", "[checkpoint] could not save on the way out"): "an error said from the engine",
    ("core.SBI.training_checkpoint", "[checkpoint] {} other checkpoint(s) exist"): "a warning said from the engine",
}
# How the engine's lines the window reads begin.
_PREFIXES = ("[fit]", "[checkpoint]", "[patho]", "[chi]", "[mem]")


class _Clock:
    """The window's clock in a test: a local time with its zone, moved by hand."""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def set(self, moment: datetime) -> None:
        self.now = moment


def _at(day, hour, minute, second=0) -> datetime:
    """A moment of October 2026 in the test's local time (the 4th is a Sunday)."""
    return datetime(2026, 10, day, hour, minute, second, tzinfo=_ZONE)


def _training(**changed) -> RunSpec:
    return RunSpec(**{"word": "Training", "place": ("inference", "training"), "name": "retrain_3",
                      "stages": _TRAINING, "simulation_set": True, "patience": 20, "epoch_limit": 300,
                      "facts": {"model": "NADROWSKI", "observation": "chi", "narrowed": False, "network": (256, 10),
                                "memory_limit": 6.0}, **changed})


def _load() -> RunSpec:
    """A run that only loads a record, as the prior's page describes one."""
    return RunSpec("Prior load", ("inference", "prior"), stop_button="Stop loading", loads_only=True,
                   keeps_files=False, stop_asks=False)


def _feed(run, *texts_, level="info"):
    run.feed([(text, level) for text in texts_])


def _epoch(epoch, since=0, took=3780.0, training=-3.0, validation=-3.1) -> str:
    """The window's literal copy of an epoch's record, as the training call writes it."""
    return (f"[fit] epoch {epoch}: training loss {training:.6g}, validation loss {validation:.6g}, took {took:.1f} s, "
            f"epochs since the best {since}")


def _frame(done, total=10_000, desc=SIMULATION_ROW, row=0):
    """A progress row as the window parses a frame of the engine's bar."""
    from core.gui import vt
    pct = int(100 * done / total) if total else 0
    raw = f"{desc}: {pct:3d}%|{'#' * (pct // 10):<10}| {done}/{total} [00:10<00:20,  9.70s/it]"
    return vt.parse_bar(("err", row), row, raw)


def _status(text, row=1):
    """A status row as the window parses one the training library prints over itself, its counter of epochs."""
    from core.gui import vt
    return vt.parse_bar(("out", row), row, f" {text}")


def _status_line(run) -> list:
    return [(word.name, word.marker, word.word) for word in run.stage_line()]


def _simulating(run, clock, *, writing=True):
    """A training brought to its simulation: the budget, the rotation, the set's first line and its first frame."""
    _feed(run, _LINES["budget"], _LINES["rotation_begins"])
    clock.advance(90)
    _feed(run, _LINES["rotation_done"])
    clock.advance(9)
    if writing:
        _feed(run, _LINES["simulation_writing"])
    run.rows([_frame(0)])


def _fitting(run, clock):
    """A training brought to its fit: simulated, the set complete."""
    _simulating(run, clock)
    clock.advance(60)
    run.rows([_frame(10_000)])
    _feed(run, _LINES["simulation_complete"])


def _resumed(run, clock, progress="best validation loss -3.412, 2 passes since the best", passes=40):
    """A kept training resumed and brought to its fit: its resume line, its complete simulation set read back (four
    minutes) with the rotation it carries."""
    _feed(run, f"[fit] resuming the kept training 20261011T121000 after pass {passes} ({progress}); its simulation "
               f"set is read back first.", _LINES["budget"],
          _LINES["rotation_reused"].replace("batches)", "batches — COMPLETE, so generation will be skipped)"),
          f"[checkpoint] resuming at batch 10000/10000 from {_FOLDER} (reusing the stored rotation V)")
    clock.advance(240)
    run.rows([_frame(10_000)])
    _feed(run, _LINES["simulation_complete"])


# ── the table ──
def _templates(node) -> list:
    """Every text a string expression can give, "{}" where a value is put in: a literal, a formatted string, a sum
    of them, either branch of a conditional; anything else in a sum is a value put in."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(part.value if isinstance(part, ast.Constant) else "{}" for part in node.values)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [a + b for a in _templates(node.left) for b in _templates(node.right)]
    if isinstance(node, ast.IfExp):
        return _templates(node.body) + _templates(node.orelse)
    return ["{}"]


def _stringy(node) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _stringy(node.left) or _stringy(node.right)
    if isinstance(node, ast.IfExp):
        return _stringy(node.body) or _stringy(node.orelse)
    return False


def _strings(tree) -> list:
    """``(line, text)`` for every string a module writes, wherever it stands (an argument, an assignment, a value
    returned), read whole: a sum and a conditional of strings once, each of its texts; the docstrings left out."""
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)}
    inside = {id(part) for node in ast.walk(tree) if _stringy(node) and not isinstance(node, ast.Constant)
              for part in ast.walk(node) if part is not node}
    return [(node.lineno, text) for node in ast.walk(tree)
            if _stringy(node) and id(node) not in inside and id(node) not in docstrings
            for text in _templates(node)]


def _source(module: str) -> Path:
    """Where a module's source is: an engine module's by its path, never imported; the training library's by its
    spec, which imports the packages above it."""
    path = REPO / (module.replace(".", "/") + ".py")
    return path if path.is_file() else Path(importlib.util.find_spec(module).origin)


def _holds(text: str, form: str) -> bool:
    """Whether ``text``, a string the engine's source writes, is a row's line whole: the form's fixed words exactly,
    each whole or real number one value put in (``{}``, or a ``%`` code), any other placeholder any text, from the
    first character to the last."""
    one = r"(?:\{\}|%[-+ #0-9.]*[a-zA-Z])"
    parts = []
    for part in re.split(r"(<\w+>)", form):
        named = re.fullmatch(r"<(\w+)>", part)
        if named is None:
            parts.append(re.escape(part))
        elif named.group(1) in readings._WHOLES | readings._REALS:
            parts.append(one)
        else:
            parts.append(r".*?")
    return re.fullmatch("".join(parts), text.lstrip(), re.DOTALL) is not None


def _calls(tree) -> list:
    """``(name, text)`` for every string a call is handed: the called function's own name ("info", "warning",
    "warn", "print") and each text the string can give."""
    return [(node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""), text)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            for arg in node.args if _stringy(arg) for text in _templates(arg)]


# What a row's message may be, by the call that writes its line: a logger's warning or a Python warning, an error,
# a line of information (none, or one of the two that say a training is not saved as it goes), a print (none).
_MESSAGES = {"warning": {"warning"}, "warn": {"warning"}, "error": {"error"}, "info": {None, "information"},
             "print": {None}}
# The rows whose line is built before the call that writes it (assigned, or returned by a function), so no call is
# read for them, each with the message it has: the window's own line of its text file is handed on as a warning.
_BUILT_FIRST = {"fit_epoch": None, "chi_total": None, "chi_none": None, "mem": None, "mem_unavailable": None,
                "text_file_broken": "warning"}


def _formatted(name):
    """For a row whose line the engine builds with a function of its own: that function's line, the values it was
    handed, and the window's literal copy where it keeps one."""
    if name == "fit_epoch":
        from core.SBI import train
        values = {"epoch": 12, "training_loss": -3.41274, "validation_loss": -3.39812, "seconds": 3541.2, "since": 0}
        return train.fit_epoch_line(12, -3.41274, -3.39812, 3541.2, 0), values, _LINES["fit_epoch"]
    from core.SBI.pipeline import _BatchProbeLedger
    if name == "chi_total":
        scope = ("only the batches this process generated; the cache's first 237 batches were committed without "
                 "per-batch counts")
        line = _BatchProbeLedger([(12, 640), (30, 640), (0, 512)], 237).summary_line(scope)
        return line, {"masked": 42, "total": 1792, "share": 2.3, "batches": 3, "scope": scope, "low": 0.0,
                      "high": 4.7, "median": 1.9}, None
    scope = "this process only (no simulation cache)"
    return _BatchProbeLedger().summary_line(scope), {"scope": scope}, None


def _resolve(dotted: str):
    """The object a dotted path names: the longest module of it imported, the rest read as attributes."""
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            found = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            found = getattr(found, name)
        return found
    raise AssertionError(f"nothing is named {dotted}")


@pytest.mark.parametrize("name", (*readings.NAMES, None), ids=lambda name: name or "every-engine-line")
def test_every_reading_matches_the_line_the_engine_writes(name):
    """Each row of the window's table is held to the engine: its line's whole shape (its fixed words exactly, a
    number's place one value, from the first character to the last) is a string the module it names writes, read
    off that module's syntax tree, so a line reworded in the engine, or given more words, fails here; its message is
    what the call that writes the line makes it (a warning, an error, information or none); a line of its form, as
    the engine writes it, is read by that row and by no other; and a row whose line the engine builds with a
    function of its own reads that function's line, with the values the function was handed (an epoch's record
    beside the window's literal copy of it).

    And the other way round (the last case): every string under core/ but the window's own folder that begins as
    the run's lines do ([fit], [checkpoint], [patho], [chi], [mem]), as a call's argument, an assignment or a value
    returned, is the line of a row of the module that writes it, or is named on this test's short list of lines the
    window reads as text, each with its reason. So a line added to the engine with neither fails here. The scanner
    and the matcher are tried on a snippet first, and every entry of the list must still name a line."""
    if name is not None:
        row = next(r for r in readings.ROWS if r.name == name)
        tree = ast.parse(_source(row.module).read_text(encoding="utf-8"))
        assert any(_holds(text, row.form) for _, text in _strings(tree)), (
            f"{row.module} no longer writes the line of {name}: {row.form!r}")
        called = {call for call, text in _calls(tree) if _holds(text, row.form)}
        if name in _BUILT_FIRST:
            assert not called and row.message == _BUILT_FIRST[name], (name, sorted(called), row.message)
        else:
            assert called and all(row.message in _MESSAGES.get(call, ()) for call in called), (
                f"{name} is said by {sorted(called)}; its message is {row.message!r}")
        line = _LINES[name]
        reading = readings.read(line, "info")
        assert reading is not None and reading.name == name, (name, reading)
        assert [r.name for r in readings.ROWS if r.pattern.fullmatch(line.strip())] == [name]
        if row.formatter is not None:
            assert callable(_resolve(row.formatter)), row.formatter
            built, given, literal = _formatted(name)
            if literal is not None:
                assert built == literal, "the window's copy of the line is no longer the engine's"
            reading = readings.read(built, "info")
            assert reading is not None and reading.name == name, built
            assert reading.values == {**reading.values, **given}, (reading.values, given)
        return

    snippet = ast.parse("\n".join([
        'def f(x):',
        '    """[fit] a docstring is no line."""',                             # left out
        '    log.info(f"[fit] saved after pass {x}.")',                          # 3
        '    line = f"[mem] {x}: peak " + ("a" if x else "b")',                 # 4: two texts
        '    return ("[chi] masked: " + ", ".join(x) + "; done")',              # 5: a call put in
        'print("plain", f"{x}")',                                               # 6, 6
    ]))
    assert sorted(_strings(snippet)) == sorted([
        (3, "[fit] saved after pass {}."), (4, "[mem] {}: peak a"), (4, "[mem] {}: peak b"),
        (5, "[chi] masked: {}; done"), (6, "plain"), (6, "{}")]), _strings(snippet)
    form = "[fit] saved after pass <passes>."
    assert _holds("[fit] saved after pass {}.", form), "a string that holds the form"
    assert not _holds("[fit] saved after pass {}. Kept.", form), "words appended"
    assert not _holds("[fit] saved after epoch {}.", form), "a word changed"
    seen, unread = [], []
    for path in sorted((REPO / "core").rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        if relative.startswith("core/gui/"):
            continue                     # the window writes none of these lines; its table holds their forms
        module = relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        for line, text in _strings(ast.parse(path.read_text(encoding="utf-8"))):
            if not text.lstrip().startswith(_PREFIXES):
                continue
            seen.append(text)
            if any(row.module == module and _holds(text, row.form) for row in readings.ROWS):
                continue
            if any(module == owner and text.lstrip().startswith(start) for owner, start in _TEXT_ONLY):
                continue
            unread.append(f"{relative}:{line}: {text}")
    assert len(seen) >= len([row for row in readings.ROWS if row.form.startswith(_PREFIXES)]), seen
    assert not unread, ("lines the engine writes that no row of the window's table reads and the list of lines read "
                        "as text does not name:\n" + "\n".join(unread))
    for owner, start in _TEXT_ONLY:
        written = [text for _, text in _strings(ast.parse(_source(owner).read_text(encoding="utf-8")))]
        assert any(text.lstrip().startswith(start) for text in written), f"{owner} no longer writes {start!r}"


def test_the_engine_words_the_model_reads_beside_its_table_are_still_the_engines():
    """Four more copies of the engine's words, read by the model outside its table, are held to the engine as its
    rows are, off the syntax tree of the module that writes them: the name of a training batch inside a line, from
    which four rows read their batch; the description of the progress row that counts a training's simulation; the
    training library's counter of the epochs it has trained, the fit's only sign when the library writes no record
    per epoch (read from the library's installed source); and the epochs since the best that a resumed training's
    line gives, from which its end is counted before its first record."""
    pipeline = ast.parse(_source("core.SBI.pipeline").read_text(encoding="utf-8"))
    tags = [text for _, text in _strings(pipeline) if text.startswith("training batch {}/{} [")]
    assert tags, "core.SBI.pipeline no longer names a training batch as the window reads it"
    for tag in tags:
        assert re.fullmatch(readings._BATCH_TAG, tag.replace("{}", "7")), tag
    descriptions = [keyword.value.value for node in ast.walk(pipeline) if isinstance(node, ast.Call)
                    for keyword in node.keywords if keyword.arg == "desc" and isinstance(keyword.value, ast.Constant)]
    assert SIMULATION_ROW in descriptions, descriptions
    library = ast.parse(_source("sbi.inference.trainers.base").read_text(encoding="utf-8"))
    assert any(model._COUNTER.fullmatch(text.replace("{}", "7")) for _, text in _strings(library)), (
        "the training library no longer writes its counter of epochs as the window reads it")
    resuming = next(row.form for row in readings.ROWS if row.name == "fit_resuming")
    fit_state = ast.parse(_source("core.SBI.fit_state").read_text(encoding="utf-8"))
    assert any(_holds(text, resuming) and model._SINCE.search(text.replace("{}", "7"))
               for _, text in _strings(fit_state)), "a resumed training no longer says its epochs since the best so"


def test_a_line_no_row_matches_is_text_and_never_an_error():
    """A line no row matches whole is read as nothing: the start of a row's line, a row's line with more after it,
    and words the engine never writes. Fed to the model, any line at all, at any level but a warning or an error,
    changes no stage, adds no epoch and raises no message; a print on the error stream is no message either."""
    for text in ("[fit] epoch 12", _LINES["fit_saved"] + " Again.", "[budget]", "", "   ", "\x1b[2K",
                 "Training batch COUNT overridden: 40 batches (config default 10000)",
                 "[fit] epoch twelve: training loss -3, validation loss -3, took 1 s, epochs since the best 0",
                 "x" * 10_000):
        assert readings.read(text, "info") is None, text
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock)
    before = _status_line(run)
    _feed(run, "[fit] epoch 12", "Writing fdt record 20261008T190211 at C:\\x", "\x1b[2K", "", level="info")
    _feed(run, _LINES["patho_batch"], "library: rejection: a print on the error stream", level="err")
    assert (_status_line(run), run.epochs(), run.messages(), run.state) == (before, (), (), "running")


def test_the_kept_trainings_lines_are_never_read_as_an_epoch():
    """The lines of the training kept as it goes say "pass" and hold numbers, and share the record's opening; none
    is read as an epoch. A window stop taken at a pass's save is said before that pass's record, which the training
    then says once while it handles the stop: the epochs read and the kept training's passes agree."""
    kept = [name for name in readings.NAMES
            if name.startswith("fit_") and name not in ("fit_epoch", "fit_saving_off", "fit_not_kept")]
    assert len(kept) == 17
    for name in kept:
        assert readings.read(_LINES[name], "info").name == name
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock)
    _fitting(run, clock)
    _feed(run, *(_LINES[name] for name in kept))
    assert run.epochs() == () and run.stage == "Fitting"
    for epoch in (1, 2):
        clock.advance(3780)
        _feed(run, f"[fit] saved after pass {epoch}.", _epoch(epoch, since=0))
    clock.advance(3780)                             # the stop is taken at the third save's line, before it is said
    _feed(run, _epoch(3, since=1), "[fit] the training stopped after pass 3 and is kept as fit 20261011T121000 "
                                   "(stopped on purpose).")
    note = SimpleNamespace(id="20261011T121000", passes=3, resumed=False, saved_here=True, processes=1)
    run.ended("stopped", kept_fit=note)
    assert [record.epoch for record in run.epochs()] == [1, 2, 3] and run.epochs()[-1].epoch == note.passes
    assert run.kept_fit is note
    assert ("Training saved", "after epoch 3", "Fine") in [tuple(vars(row).values()) for row in run.health()]


def test_a_training_reads_through_its_stages_with_their_words():
    """A training's stage line: each stage "To come" until the line that begins it, "Now" while it runs, then its
    words when done: the rotation "Done", or "Reused from the saved simulation set" when a continued set carries it,
    "The parent posterior's" for a narrowing round, "Off" when the page says the training is not rotated or the
    rotation is passed over; the simulation "Done, 20,480,000 simulations", or "Not needed: the saved simulations
    are used" when the set was complete; the fit "Now, about 63 minutes per epoch" once it has a pace, from its
    records or, when the training library writes none, from the gaps between its counter's values. Saving begins at
    the record whose count since the best reaches the patience, at the record past the epoch limit, or at the
    training library's own word that it converged."""
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock)
    assert (run.state, run.stage) == ("starting", None)
    assert _status_line(run) == [("Rotation", "later", "To come"), ("Simulation", "later", "To come"),
                                 ("Fitting", "later", "To come"), ("Saving", "later", "To come")]
    _feed(run, _LINES["budget"])
    assert (run.state, run.stage, run.began_after_first()) == ("running", None, False)
    _feed(run, _LINES["rotation_begins"])
    assert run.stage == "Rotation" and _status_line(run)[0] == ("Rotation", "now", "Now")
    _feed(run, _LINES["rotation_done"])
    assert _status_line(run)[0] == ("Rotation", "done", "Done")
    _feed(run, _LINES["simulation_writing"])
    run.rows([_frame(0)])
    assert run.stage == "Simulation" and run.began_after_first() and run.simulation_folder() == _FOLDER
    assert _status_line(run)[:2] == [("Rotation", "done", "Done"), ("Simulation", "now", "Now")]
    run.rows([_frame(10_000)])
    _feed(run, _LINES["simulation_complete"])
    assert run.stage == "Fitting" and run.batches_done() == 10_000
    assert _status_line(run)[1:3] == [("Simulation", "done", "Done, 20,480,000 simulations"),
                                      ("Fitting", "now", "Now")]
    _feed(run, _epoch(1, took=3780.0))
    assert _status_line(run)[2] == ("Fitting", "now", "Now, about 63 minutes per epoch")
    _feed(run, *(_epoch(n, since=max(0, n - 3)) for n in range(2, 24)))
    assert run.stage == "Saving" and run.best_epoch() == 3, "the count since the best reached the patience"
    assert _status_line(run)[2:] == [("Fitting", "done", "Done"), ("Saving", "now", "Now")]
    run.ended("finished")
    assert _status_line(run)[3] == ("Saving", "done", "Done") and run.axis() is None

    unrotated = RunModel(_training(facts={"rotated": False}), clock=clock)
    assert _status_line(unrotated)[0] == ("Rotation", "none", "Off")
    passed = RunModel(_training(), clock=clock)
    _feed(passed, _LINES["budget"], _LINES["simulation_writing"])
    assert _status_line(passed)[:2] == [("Rotation", "none", "Off"), ("Simulation", "now", "Now")]

    continued = RunModel(_training(), clock=clock)
    _feed(continued, _LINES["budget"], _LINES["rotation_reused"], _LINES["simulation_resuming"])
    assert _status_line(continued)[:2] == [("Rotation", "done", "Reused from the saved simulation set"),
                                           ("Simulation", "now", "Now")]
    assert continued.batches_done() == 236
    narrowing = RunModel(_training(), clock=clock)
    _feed(narrowing, _LINES["budget"], _LINES["rotation_parent"])
    assert _status_line(narrowing)[0] == ("Rotation", "done", "The parent posterior's")
    complete = RunModel(_training(), clock=clock)
    _feed(complete, _LINES["budget"], _LINES["rotation_reused"].replace("batches)", "batches — COMPLETE, so "
                                                                                    "generation will be skipped)"),
          f"[checkpoint] resuming at batch 10000/10000 from {_FOLDER} (reusing the stored rotation V)",
          _LINES["simulation_complete"])
    assert _status_line(complete)[1] == ("Simulation", "done", "Not needed: the saved simulations are used")

    limited = RunModel(_training(epoch_limit=5), clock=clock)
    _fitting(limited, clock)
    _feed(limited, *(_epoch(n, since=0) for n in range(1, 6)))
    assert limited.stage == "Fitting"
    _feed(limited, _epoch(6, since=0))
    assert limited.stage == "Saving", "the record past the epoch limit"
    converged = RunModel(_training(patience=None), clock=clock)
    _fitting(converged, clock)
    converged.rows([_status("Training neural network. Epochs trained: 1")])
    assert converged.stage == "Fitting" and _status_line(converged)[2] == ("Fitting", "now", "Now")
    clock.advance(3780)                              # the library's counter, with no record: its pace from the gaps
    converged.rows([_status("Training neural network. Epochs trained: 2")])
    assert _status_line(converged)[2] == ("Fitting", "now", "Now, about 63 minutes per epoch")
    converged.rows([_status("Neural network successfully converged after 131 epochs.")])
    assert converged.stage == "Saving"

    sweep = RunModel(RunSpec("Sweep study", ("temperature", "sweep"), name="sweep",
                             stages=(Stage("S sweep", ("sweep_phase",), "sweep_point_b"),
                                     Stage("T_a/T sweep", ("sweep_phase",), "sweep_point_b"))), clock=clock)
    _feed(sweep, "--- Phase A (s sweep): spontaneous PSD + omega_0 detection ---", "  [A 1/8] s=0 (temp=1)",
          "--- Phase B (s sweep): forced response on common grid ---", "  [B 1/8] s=0")
    assert _status_line(sweep) == [("S sweep", "now", "Now"), ("T_a/T sweep", "later", "To come")]
    assert (sweep.axis().end_by, sweep.axis().unknown) == (None, "No end time yet: the first steps are being timed.")
    _feed(sweep, _LINES["sweep_phase"])
    assert _status_line(sweep) == [("S sweep", "done", "Done"), ("T_a/T sweep", "now", "Now")]


def test_a_short_counted_stage_gets_an_end_time_from_a_quarter_of_its_count():
    """A counted stage that is not the simulation takes its own pace once a quarter of its count is timed: a sweep of
    eight points has a firm end from its third point and not before. The second of two sweeps counted alike is
    counted, and timed, from its own phase line."""
    clock = _Clock(_at(8, 19, 2, 11))
    sweep = RunModel(RunSpec("Sweep study", ("temperature", "sweep"), name="sweep",
                             stages=(Stage("S sweep", ("sweep_phase",), "sweep_point_b"),
                                     Stage("T_a/T sweep", ("sweep_phase",), "sweep_point_b"))), clock=clock)
    for param, other, step in (("s", "temp=1", 1800), ("temp", "s=0", 900)):
        _feed(sweep, f"--- Phase A ({param} sweep): spontaneous PSD + omega_0 detection ---")
        began = clock()
        for i in range(1, 9):
            clock.advance(600)
            _feed(sweep, f"  [A {i}/8] {param}={i} ({other})")
        _feed(sweep, f"--- Phase B ({param} sweep): forced response on common grid ---")
        for i in range(1, 4):
            clock.advance(step)
            _feed(sweep, f"  [B {i}/8] {param}={i}")
            axis = sweep.axis()
            if i < 3:
                assert (axis.start, axis.end_by, axis.unknown) == (
                    began, None, "No end time yet: the first steps are being timed."), (param, i)
        assert (axis.start, axis.end_from, axis.end_by, axis.unknown) == (
            began, clock() + timedelta(seconds=5 * step), clock() + timedelta(seconds=5 * step), ""), param


def test_a_run_with_no_stages_says_and_times_the_count_of_its_driving_row():
    """A run with no stages whose progress row counts says its count, "<done> of <total>", and has a time axis from
    its start by the simulation's rule on that row: no end until enough steps are timed, then a firm one at its own
    pace. A count other than the simulation's is timed over a quarter of it, two steps at the least and twenty, the
    simulation's, at the most."""
    clock = _Clock(_at(8, 19, 2, 11))
    check = RunModel(RunSpec("Check", ("inference", "checks"), name="sbc_1"), clock=clock)
    assert (check.sentence(), check.axis()) == ("Check sbc_1 is starting.", None)
    clock.advance(5)
    check.rows([_frame(0, 40, desc="Calibrating")])
    assert check.sentence() == "Check sbc_1 is running: 0 of 40."
    axis = check.axis()
    assert (axis.start, axis.end_by, axis.unknown, axis.follows) == (
        _at(8, 19, 2, 11), None, "No end time yet: the first steps are being timed.", "")
    for done in range(1, 11):
        clock.advance(30)
        check.rows([_frame(done, 40, desc="Calibrating")])
    assert check.sentence() == "Check sbc_1 is running: 10 of 40."
    axis = check.axis()
    assert axis.end_from == axis.end_by == clock() + timedelta(seconds=30 * 30)
    for total, timed in ((5, 2), (40, 10), (100, 20)):
        counted = RunModel(RunSpec("Check", ("inference", "checks"), name="sbc_2"), clock=clock)
        for done in range(timed + 1):
            counted.rows([_frame(done, total, desc="Calibrating")])
            assert (counted.axis().end_by is None) == (done < timed), (total, done)
            clock.advance(30)


def test_once_a_run_has_ended_nothing_begins_or_moves_a_stage():
    """An end is the last word on the stage line. A finished run's stages all read done but those passed over,
    among them the saving after a fit the training library counted without a record and ended at its limit
    without a word; and a line, a row or a record that would begin, end or leave out a stage, read after any end,
    moves nothing and raises nothing."""
    clock = _Clock(_at(8, 19, 2, 11))
    finished = RunModel(_training(patience=None), clock=clock)
    _fitting(finished, clock)
    finished.rows([_status("Training neural network. Epochs trained: 300")])
    finished.ended("finished")
    before = (finished.stage, _status_line(finished))
    _feed(finished, "Neural network successfully converged after 131 epochs.", _LINES["simulation_complete"])
    finished.rows([_status("Neural network successfully converged after 131 epochs.")])
    assert (finished.stage, _status_line(finished)) == before, "nothing moves after the end"
    assert before == ("Fitting", [("Rotation", "done", "Done"), ("Simulation", "done", "Done, 20,480,000 simulations"),
                                  ("Fitting", "done", "Done"), ("Saving", "done", "Done")])
    for outcome, word in (("stopped", "Stopped"), ("failed", "Failed")):
        cut = RunModel(_training(), clock=clock)
        _simulating(cut, clock)
        cut.ended(outcome)
        _feed(cut, _LINES["simulation_complete"], _epoch(1, since=20))
        assert (cut.stage, _status_line(cut)) == ("Simulation", [
            ("Rotation", "done", "Done"), ("Simulation", "half", word), ("Fitting", "later", "To come"),
            ("Saving", "later", "To come")]), outcome
    cell = RunModel(RunSpec("Measurement", ("temperature", "cell"), name="cell_a", stages=_CELL), clock=clock)
    _feed(cell, "Writing fdt record 20261008T190211 at C:\\x")
    cell.ended("stopped")
    _feed(cell, _LINES["fdt_sanity_skipped"], _LINES["fdt_campaign_spontaneous"])
    assert _status_line(cell) == [("Checks", "half", "Stopped"), ("Spontaneous", "later", "To come"),
                                  ("Driven", "later", "To come")]


def test_a_stop_asked_while_starting_still_lets_the_first_stage_begin():
    """A stop asked for before the run's first line (the window closed while the run starts) does not keep its first
    stage from beginning with that line, so the stage the stop cuts short is the one drawn stopped."""
    clock = _Clock(_at(8, 19, 2, 11))
    cell = RunModel(RunSpec("Measurement", ("temperature", "cell"), name="cell_a", stages=_CELL), clock=clock)
    cell.stopping(True)
    assert (cell.state, cell.stage) == ("stopping", None)
    _feed(cell, "Writing fdt record 20261008T190211 at C:\\x")
    assert (cell.state, cell.stage, cell.sentence()) == (
        "stopping", "Checks", "Stopping: takes effect at the run's next line of output.")
    cell.ended("stopped")
    assert _status_line(cell) == [("Checks", "half", "Stopped"), ("Spontaneous", "later", "To come"),
                                  ("Driven", "later", "To come")]


def test_each_state_says_its_sentence():
    """One sentence says where the run stands, in the words the page uses: starting, the rotation, the simulation
    with its pace and its end, the fit before its first epoch (with the pace of an earlier training of this size
    when there is one), the fit with its epoch of the expected ones, past the estimate, and the saving. A time
    within seven days carries its weekday and no year, any other its year. A run of another kind says its stage, and
    one that counts for itself says its count; a stop asked for and an end each say theirs, and the stage an end
    cuts short is drawn half done, with the end's word."""
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock)
    assert run.sentence() == "Training retrain_3 is starting."
    _feed(run, _LINES["budget"], _LINES["rotation_begins"])
    assert run.sentence() == "The rotation is being computed before any simulation."
    clock.set(_at(8, 19, 3, 41))
    _feed(run, _LINES["rotation_done"])
    assert run.sentence() == "Training retrain_3 is starting."
    clock.set(_at(8, 19, 3, 50))
    _feed(run, _LINES["simulation_writing"])
    run.rows([_frame(0)])
    assert run.sentence() == "Simulating: batch 0 of 10,000."
    clock.advance(57_715)                            # 9.7 s a batch from here
    run.rows([_frame(5_950)])
    clock.advance(1_940)
    run.rows([_frame(6_150)])
    assert clock() == _at(9, 11, 38, 5)
    assert run.sentence() == ("Simulating: batch 6,150 of 10,000, 9.7 s per batch; the simulations should be done by "
                              "Fri 9 Oct, 22:00.")
    run.stopping(True)
    assert (run.state, run.sentence()) == ("stopping", "Stopping: the finished batches are being saved.")
    run.stopping(False)
    assert run.state == "running" and run.sentence().startswith("Simulating: batch 6,150 of 10,000")

    clock = _Clock(_at(8, 19, 3, 50))
    slow = RunModel(_training(), clock=clock)
    _feed(slow, _LINES["budget"], _LINES["simulation_writing"])
    slow.rows([_frame(0)])
    clock.advance(40_000)                            # 200 s a batch: the end is more than a week away
    slow.rows([_frame(200)])
    assert slow.sentence() == ("Simulating: batch 200 of 10,000, 200.0 s per batch; the simulations should be done "
                               "by 31 Oct 2026, 22:37.")

    def estimate(per, size):
        return (3780.0, "the last training of this size, retrain_aug, took about 63 minutes per epoch") \
            if per == "epoch" else None

    for given, said, bound in ((None, "", "."), (estimate, " At this size an epoch last took about 63 minutes.",
                                                 ", up to about 63 minutes.")):
        clock = _Clock(_at(8, 19, 2, 11))
        fitting = RunModel(_training(), clock=clock, estimate=given)
        _fitting(fitting, clock)
        assert fitting.sentence() == ("The simulations are done. The fit has started; its first epoch has not "
                                      "finished." + said)
        fitting.stopping(True)                       # no epoch measured: the bound is the history's, or none
        assert fitting.sentence() == "Stopping: takes effect when this epoch ends" + bound

    for expected, of in ((_EXPECTED, " of about 130"), (None, "")):
        clock = _Clock(_at(8, 19, 2, 11))
        fitting = RunModel(_training(), clock=clock, expected=expected)
        _fitting(fitting, clock)
        _feed(fitting, *(_epoch(n, since=n % 5) for n in range(1, 63)))
        clock.set(_at(11, 12, 10, 30))
        assert fitting.sentence() == f"Now: Sun 11 Oct, 12:10, running 65 hours, epoch 62{of}."
        fitting.stopping(True)
        assert fitting.sentence() == "Stopping: takes effect when this epoch ends, up to about 63 minutes."
    clock = _Clock(_at(8, 19, 2, 11))
    young = RunModel(_training(), clock=clock)
    _fitting(young, clock)
    _feed(young, _epoch(1, took=600.0))
    clock.set(_at(8, 19, 47, 50))
    assert young.sentence() == "Now: Thu 8 Oct, 19:47, running 45 minutes, epoch 1."

    clock = _Clock(_at(13, 9, 3, 51))                # the fit begins at 09:06:30, after the simulation's lines
    past = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _fitting(past, clock)
    assert clock() == _at(13, 9, 6, 30)
    for n in range(1, 153):
        took = 600.0 if n <= 142 else 3780.0
        clock.advance(took)
        _feed(past, _epoch(n, since=n % 3, took=took))
        if n == 130:
            assert past.sentence().endswith("epoch 130 of about 130.")
        if n == 140:                                 # past the estimate, still inside its range
            assert clock() == _at(14, 8, 26, 30)
            assert past.sentence() == ("Epoch 140. The estimate was about 130. The fit ends by itself 20 epochs after "
                                       "its last improvement: not before Wed 14 Oct, 11:26.")
            axis = past.axis()                       # the range would end at epoch 150; it cannot end before 158
            assert axis.end_from == axis.end_by == clock() + timedelta(seconds=18 * 600) > axis.now, "not late yet"
    assert clock() == _at(14, 19, 16, 30)
    assert past.sentence() == ("Epoch 152. The estimate was about 130. The fit ends by itself 20 epochs after its last "
                               "improvement: not before Thu 15 Oct, 14:10.")
    _feed(past, _epoch(153, since=20))
    assert past.sentence() == "The fit ended after 153 epochs. The posterior is being saved."
    converged = RunModel(_training(), clock=clock)
    _fitting(converged, clock)
    converged.feed([("Neural network successfully converged after 131 epochs.", "info")])
    assert converged.sentence() == "The fit ended after 131 epochs. The posterior is being saved."

    cell = RunModel(RunSpec("Measurement", ("temperature", "cell"), name="cell_a",
                            stages=(Stage("Checks"), Stage("Spontaneous", ("fdt_campaign_spontaneous",)),
                                    Stage("Driven", ("fdt_campaign_driven",)))), clock=clock)
    assert cell.sentence() == "Measurement cell_a is starting."
    _feed(cell, "Writing fdt record 20261008T190211 at C:\\x")
    assert cell.sentence() == "Measurement cell_a is running: Checks."
    _feed(cell, _LINES["fdt_sanity_skipped"], _LINES["fdt_campaign_spontaneous"])
    assert cell.sentence() == "Measurement cell_a is running: Spontaneous."
    assert _status_line(cell) == [("Spontaneous", "now", "Now"), ("Driven", "later", "To come")]
    live = RunModel(RunSpec("Live simulation", ("live", None)), clock=clock)
    live.set_count(1_200, 5_000, "samples")
    assert live.sentence() == "Live simulation is running: 1,200 of 5,000 samples."
    card = RunModel(RunSpec("Card test", ("inference", "checks")), clock=clock)
    card.set_stage("Simulation")
    assert (card.stage, card.sentence()) == ("Simulation", "Card test is running: Simulation.")

    for outcome, said, simulation in (
            ("finished", "Training retrain_3 finished.", ("done", "Done, 20,480,000 simulations")),
            ("stopped", "Training retrain_3 was stopped.", ("half", "Stopped")),
            ("failed", "Training retrain_3 failed.", ("half", "Failed")),
            ("not seen to end", "Training retrain_3 was not seen to end.", ("half", "Not seen to end"))):
        ending = RunModel(_training(), clock=clock)
        _feed(ending, _LINES["budget"], _LINES["simulation_writing"])
        ending.ended(outcome)
        assert (ending.state, ending.outcome, ending.sentence()) == ("ended", outcome, said)
        assert _status_line(ending)[:2] == [("Rotation", "none", "Off"), ("Simulation", *simulation)], outcome
    with pytest.raises(ValueError):
        RunModel(_training(), clock=clock).ended("vanished")
    assert model.IDLE.state == "idle" and not model.IDLE.live and not model.IDLE.unacknowledged_end
    assert [model.RunStatus(state=state).live for state in model.STATES] == [
        False, True, True, True, True, False, False, False]
    assert model.RunStatus(state="ended", outcome="failed").unacknowledged_end
    assert not model.RunStatus(state="ended", acknowledged=True).unacknowledged_end
    assert model.RunStatus(state="not_seen").unacknowledged_end
    spec = _training()
    called = spec.with_call(len, ["a"], {"b": 1})
    assert (called.fn, called.args, called.kwargs, called.name, spec.fn) == (len, ("a",), {"b": 1}, "retrain_3", None)


def test_the_simulations_end_is_firm_and_paced_on_the_last_two_hundred_batches():
    """The simulation's end is firm: earliest and latest both now plus the batches left at the pace of the newest
    200 batches of this try, recomputed at every look, so it is never late. Until 20 batches are timed the pace is
    the history's for this size, and with none no end is known. A continued set times from where it resumed, and
    counts the saves it passes from there. Every moment carries its zone, so the time axis can draw them."""
    from core.gui.widgets.time_axis import axis_layout
    clock = _Clock(_at(8, 19, 3, 50))
    run = RunModel(_training(), clock=clock)
    _feed(run, _LINES["budget"], _LINES["simulation_writing"])
    run.rows([_frame(0)])
    axis = run.axis()
    assert (axis.start, axis.now, axis.end_from, axis.end_by, axis.unknown) == (
        _at(8, 19, 3, 50), _at(8, 19, 3, 50), None, None, "No end time yet: the first batches are being timed.")
    clock.advance(19 * 30)
    run.rows([_frame(19)])
    assert run.axis().end_by is None, "19 batches are not yet timed"
    clock.advance(30)
    run.rows([_frame(20)])
    assert run.axis().end_by == clock() + timedelta(seconds=9_980 * 30.0)
    clock.advance(279 * 20)                          # to 299 at 20 s a batch, one of 100 s, then 200 at 9.7 s
    run.rows([_frame(299)])
    clock.advance(100)
    run.rows([_frame(300)])
    clock.advance(1_940)
    run.rows([_frame(500)])
    axis = run.axis()                                # batches 300 to 500 exactly: one more would take in the slow one
    assert axis.end_from == axis.end_by == clock() + timedelta(seconds=9_500 * 9.7) and axis.unknown == ""
    clock.advance(600)                               # no batch for ten minutes: the end moves on with the clock
    later = run.axis()
    assert later.end_by == clock() + timedelta(seconds=9_500 * 9.7) and later.end_by > later.now
    assert all(moment.tzinfo is not None for moment in (later.start, later.now, later.end_from, later.end_by))
    assert axis_layout(later.start, later.now, later.end_from, later.end_by, 600, len)["late"] is False

    asked = []

    def estimate(per, size):
        asked.append((per, size))
        return (9.7, "the last training of this size, retrain_aug, took 9.7 s per batch") if per == "batch" else None

    continued = RunModel(_training(facts={**_training().facts, "cadence": 50}), clock=clock, estimate=estimate)
    _feed(continued, _LINES["budget"], _LINES["simulation_resuming"])
    continued.rows([_frame(236)])
    assert continued.axis().end_by == clock() + timedelta(seconds=(10_000 - 236) * 9.7), "the history's pace"
    assert ("batch", {"per_batch": 2_048, "simulations": 20_480_000, "network": (256, 10)}) in asked
    clock.advance(19 * 5)
    continued.rows([_frame(255)])
    assert continued.axis().end_by == clock() + timedelta(seconds=(10_000 - 255) * 9.7)
    clock.advance(5)
    continued.rows([_frame(256)])
    assert continued.axis().end_by == clock() + timedelta(seconds=(10_000 - 256) * 5.0), "its own pace from 236"
    assert (continued.batches_done(), continued.saves_passed()) == (256, 1), "resumed at 236: the save at 250"
    assert len([a for a in asked if a[0] == "batch"]) == 1, "the history is read once for a size"
    _feed(run, _LINES["stopping"])
    assert run.axis() is not None and run.saves_passed() == 10, "a save passed at each 50 batches"


def test_the_fits_end_is_a_range_one_patience_either_side_of_the_expected_epochs():
    """The fit's end, from the median time of its newest ten epochs: it cannot end before its count since the best
    reaches the patience; with the expected epochs, the range runs from the later of that and the expected epochs
    less one patience to the expected epochs plus one patience; never past the epoch limit plus one. Before the
    first epoch the pace is the history's for this size (none: no end is known); with no expected epochs only when
    it cannot end before is said; past the range the axis is late, due at the record that closed it, or, with that
    record missing, when it should have come."""
    clock = _Clock(_at(8, 19, 2, 11))
    bare = RunModel(_training(), clock=clock)
    _fitting(bare, clock)
    began = clock()
    axis = bare.axis()
    assert (axis.start, axis.end_from, axis.end_by, axis.unknown) == (
        began, None, None, "No end time yet: the first epoch has not finished.")

    def estimate(per, size):
        return (3780.0, "the last training of this size took about 63 minutes per epoch") if per == "epoch" else None

    hour = timedelta(seconds=3780)
    first = RunModel(_training(), clock=clock, estimate=estimate, expected=_EXPECTED)
    _fitting(first, clock)
    began = clock()
    axis = first.axis()
    assert (axis.start, axis.end_from, axis.end_by, axis.unknown) == (began, began + 110 * hour, began + 150 * hour, "")
    assert axis.follows == "About 130 epochs: the last training of this kind, retrain_aug, ran 130."

    clock = _Clock(_at(12, 5, 6, 51))                # the fit begins at 05:09:30, after the simulation's lines
    alone = RunModel(_training(), clock=clock)
    _fitting(alone, clock)
    for n in range(1, 11):
        clock.advance(3780)
        _feed(alone, _epoch(n, since=min(n - 1, 3)))
    assert clock() == _at(12, 15, 39, 30)
    axis = alone.axis()
    assert (axis.start, axis.end_from, axis.end_by) == (_at(12, 5, 9, 30), _at(13, 9, 30, 30), None)
    assert axis.unknown == ("Cannot end before Tue 13 Oct, 09:30. No expected number of epochs was given, and there "
                            "is no earlier training of this kind.")
    assert axis.follows == ""

    clock = _Clock(_at(8, 19, 2, 11))
    ranged = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _fitting(ranged, clock)
    for n in range(1, 63):                           # 52 short epochs, then ten of 63 minutes
        took = 1000.0 if n <= 52 else 3780.0
        clock.advance(took)
        _feed(ranged, _epoch(n, since=4 if n == 62 else 0, took=took))
    axis = ranged.axis()
    assert (axis.end_from, axis.end_by) == (clock() + 48 * hour, clock() + 88 * hour), "epochs 110 and 150"
    for n in range(63, 101):
        clock.advance(3780)
        _feed(ranged, _epoch(n))
    axis = ranged.axis()
    assert (axis.end_from, axis.end_by) == (clock() + 20 * hour, clock() + 50 * hour), "cannot end before epoch 120"
    clock.advance(5000)
    _feed(ranged, _epoch(101, took=5000.0))          # one long epoch among the newest ten moves no median
    axis = ranged.axis()
    assert (axis.end_from, axis.end_by) == (clock() + 20 * hour, clock() + 49 * hour)

    clock = _Clock(_at(8, 19, 2, 11))
    limited = RunModel(_training(epoch_limit=60), clock=clock, expected=_EXPECTED)
    _fitting(limited, clock)
    for n in range(1, 51):
        clock.advance(3780)
        _feed(limited, _epoch(n, since=3 if n == 50 else 0))
    axis = limited.axis()
    assert axis.end_from == axis.end_by == clock() + 11 * hour, "never past epoch 61"

    clock = _Clock(_at(8, 19, 2, 11))
    paced = RunModel(_training(), clock=clock)
    _fitting(paced, clock)
    for n, took in enumerate((100.0, *[3000.0] * 5, *[4000.0] * 5), start=1):
        clock.advance(took)
        _feed(paced, _epoch(n, took=took))
    assert paced.axis().end_from == clock() + timedelta(seconds=20 * 3500), (
        "the median of the newest ten, 3,000 s and 4,000 s five times each: one epoch fewer or more moves it")

    clock = _Clock(_at(8, 19, 2, 11))
    late = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _fitting(late, clock)
    for n in range(1, 153):
        clock.advance(3780)
        _feed(late, _epoch(n, since=n % 3))
        if n == 150:
            closed = clock()
    axis = late.axis()
    assert axis.end_from == axis.end_by == closed and axis.now > axis.end_by
    from core.gui.widgets.time_axis import axis_layout
    assert axis_layout(axis.start, axis.now, axis.end_from, axis.end_by, 600, len)["late"] is True

    clock = _Clock(_at(8, 19, 2, 11))
    skipped = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _fitting(skipped, clock)
    for n in (*range(1, 150), 151, 152):             # the record of epoch 150, which closes the range, is missing
        clock.advance(3780 * (2 if n == 151 else 1))
        _feed(skipped, _epoch(n, since=n % 3))
    axis = skipped.axis()
    assert axis.end_from == axis.end_by == clock() - 2 * hour < axis.now, "due when epoch 150 should have ended"


def test_a_resumed_fit_counts_from_the_pass_it_resumed_after():
    """A kept training resumed says the pass it resumes after and its epochs since the best. Before the first record
    of this process its sentence and its end count from there, as its records then do: resumed after pass 40, two
    passes since the best, the range runs from epoch 110 to 150; with no expected epochs it cannot end before 18
    more, or 21 when no loss so far was a finite number, as before a fit's first epoch."""
    def estimate(per, size):
        return (3780.0, "the last training of this size took about 63 minutes per epoch") if per == "epoch" else None

    hour = timedelta(seconds=3780)
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock, estimate=estimate, expected=_EXPECTED)
    _resumed(run, clock)
    began = clock()
    assert run.stage == "Fitting" and run.epochs() == ()
    assert run.sentence() == "Now: Thu 8 Oct, 19:06, running 4 minutes, epoch 40 of about 130."
    axis = run.axis()
    assert (axis.start, axis.end_from, axis.end_by) == (began, began + 70 * hour, began + 110 * hour)
    for n in (41, 42):
        clock.advance(3780)
        _feed(run, _epoch(n, since=n - 38))
    assert run.sentence() == "Now: Thu 8 Oct, 21:12, running 2 hours, epoch 42 of about 130."
    axis = run.axis()
    assert (axis.start, axis.end_from, axis.end_by) == (began, clock() + 68 * hour, clock() + 108 * hour)
    for progress, least in (("best validation loss -3.412, 2 passes since the best", 18),
                            ("no finite validation loss yet", 21)):
        clock = _Clock(_at(8, 19, 2, 11))
        alone = RunModel(_training(), clock=clock, estimate=estimate)
        _resumed(alone, clock, progress)
        assert alone.sentence() == "Now: Thu 8 Oct, 19:06, running 4 minutes, epoch 40.", progress
        assert alone.axis().end_from == clock() + least * hour, progress


def test_a_resumed_fit_past_its_range_is_due_no_earlier_than_its_start_here():
    """A fit resumed past its expected range has no record, in this process, of the epoch that closed the range: its
    axis is late, due no earlier than the fit began here, never before the axis starts."""
    from core.gui.widgets.time_axis import axis_layout
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _resumed(run, clock, "best validation loss -3.412, 3 passes since the best", passes=155)
    for n in (156, 157):
        clock.advance(3780)
        _feed(run, _epoch(n, since=n - 152))
    axis = run.axis()
    assert axis.end_from == axis.end_by == axis.start == clock() - timedelta(seconds=2 * 3780) < axis.now
    assert axis_layout(axis.start, axis.now, axis.end_from, axis.end_by, 600, len)["late"] is True


def test_the_axis_names_what_follows_the_stage_running():
    """Under the simulation's axis, the fit that follows it, from the history's pace of an epoch at this size and
    the expected epochs, one patience either side, in the unit of the shorter: "Then the fit: about 5 to 7 days.";
    nothing without the pace or without the expected epochs. Under the fit's, where its expected epochs come from.
    The rotation and the saving have no axis."""
    def paced(seconds):
        return lambda per, size: (seconds, "the last training of this size took that long") if per == "epoch" else None

    for seconds, expected, patience, said in (
            (3780.0, _EXPECTED, 20, "Then the fit: about 5 to 7 days."),
            (600.0, (30, "about 30 epochs: the last training of this kind, a, ran 30"), 5,
             "Then the fit: about 4 to 6 hours."),
            (120.0, (30, "about 30 epochs: the last training of this kind, a, ran 30"), 5,
             "Then the fit: about 50 to 70 minutes.")):
        clock = _Clock(_at(8, 19, 2, 11))
        run = RunModel(_training(patience=patience), clock=clock, estimate=paced(seconds), expected=expected)
        _feed(run, _LINES["budget"], _LINES["rotation_begins"])
        assert run.axis() is None
        _feed(run, _LINES["rotation_done"], _LINES["simulation_writing"])
        run.rows([_frame(0)])
        assert run.axis().follows == said
    for estimate, expected in ((None, _EXPECTED), (paced(3780.0), None)):
        clock = _Clock(_at(8, 19, 2, 11))
        run = RunModel(_training(), clock=clock, estimate=estimate, expected=expected)
        _simulating(run, clock)
        assert run.axis().follows == ""
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock, expected=_EXPECTED)
    _fitting(run, clock)
    _feed(run, _epoch(1))
    assert run.axis().follows == "About 130 epochs: the last training of this kind, retrain_aug, ran 130."
    _feed(run, _epoch(2, since=20))
    assert run.stage == "Saving" and run.axis() is None


def _health(run) -> list:
    return [(row.measure, row.now, row.state) for row in run.health()]


def test_the_health_table_reads_its_four_measures():
    """A training's health: failed simulations as a share of those simulated ("Fine" up to 1 %), since the batch a
    continued run resumed at, or over every run of the set once the earlier ones are handed in; unusable driven
    recordings once simulation ends, not compared yet; the graphics card's largest peak of its memory with the
    limit, during simulation once the fit has begun, "Look closer" once a batch had to be split or failed whole;
    and the training saved as it goes, "Look closer" after a failed save until the next save. A run that simulates
    no training rows has none."""
    clock = _Clock(_at(8, 19, 2, 11))
    run = RunModel(_training(), clock=clock)
    assert _health(run) == [("Failed simulations", "None so far", "Fine"),
                            ("Unusable driven recordings", "Counted when simulation ends", ""),
                            ("Graphics-card memory", "Not measured yet", "")]
    _simulating(run, clock)
    _feed(run, _LINES["patho_batch"], _LINES["mem"])
    assert _health(run) == [("Failed simulations", "0.15 %", "Fine"),
                            ("Unusable driven recordings", "Counted when simulation ends", ""),
                            ("Graphics-card memory", "10.4 of 15.9 GiB, limit 6 GiB", "Fine")]
    _feed(run, _LINES["mem"].replace("10.42", "9.10"), _LINES["mem_unavailable"])
    assert _health(run)[2] == ("Graphics-card memory", "10.4 of 15.9 GiB, limit 6 GiB", "Fine"), "the largest peak"
    _feed(run, _LINES["oom_split_simulation"], level="warning")
    assert _health(run)[2][2] == "Look closer"
    _feed(run, "[patho] batch 6200: 9 new pathological trajectorie(s) -- 200,000 non-finite, 6,450 exactly constant, "
               "0 over 1e+06 in magnitude, of 12,697,600 simulated", level="warning")
    assert _health(run)[0] == ("Failed simulations", "1.63 %", "Look closer")
    _feed(run, _LINES["patho_total"], _LINES["chi_total"], _LINES["simulation_complete"])
    assert _health(run)[:2] == [("Failed simulations", "0.15 %", "Fine"),
                                ("Unusable driven recordings", "40.1 %", "Not compared")]
    assert _health(run)[2] == ("Graphics-card memory", "10.4 of 15.9 GiB, limit 6 GiB, during simulation",
                               "Look closer")
    assert len(_health(run)) == 3, "the training saved shows once it is saved"
    _feed(run, "[fit] saved after pass 40.")
    assert _health(run)[3] == ("Training saved", "after epoch 40", "Fine")
    _feed(run, "[fit] the last save failed: OSError: [Errno 28] No space left on device; the kept training stands at "
               "pass 40.", level="warning")
    assert _health(run)[3] == ("Training saved", "epoch 40; the last save failed", "Look closer")
    _feed(run, "[fit] saved after pass 42.")
    assert _health(run)[3] == ("Training saved", "after epoch 42", "Fine")

    unkept = RunModel(_training(), clock=clock)
    _fitting(unkept, clock)
    _feed(unkept, _LINES["fit_save_failed_nothing_kept"], level="warning")
    assert _health(unkept)[-1] == ("Training saved", "nothing kept yet; the last save failed", "Look closer")
    unsaved = RunModel(_training(), clock=clock)
    _feed(unsaved, _LINES["fit_not_kept"])
    _fitting(unsaved, clock)
    _feed(unsaved, _LINES["chi_none"])
    assert _health(unsaved)[1] == ("Unusable driven recordings", "None counted", "Not compared")
    assert [measure for measure, _, _ in _health(unsaved)] == [
        "Failed simulations", "Unusable driven recordings", "Graphics-card memory"]
    assert unsaved.fit_not_saved() and not run.fit_not_saved()

    continued = RunModel(_training(facts={"observation": "forced"}), clock=clock)
    _feed(continued, _LINES["budget"], _LINES["simulation_resuming"], _LINES["patho_batch"])
    assert _health(continued)[0] == ("Failed simulations", "0.15 % since batch 236", "Fine")
    assert [measure for measure, _, _ in _health(continued)] == ["Failed simulations", "Graphics-card memory"]
    continued.set_earlier(((30_000, 485_376), (1_000, 100_000)))
    assert _health(continued)[0] == ("Failed simulations", "0.38 % over 3 runs", "Fine"), "49,450 of 13,180,576"
    assert RunModel(RunSpec("Measurement", ("temperature", "cell")), clock=clock).health() == ()


def _openings(run, now) -> list:
    from core.gui.widgets.message_row import opening
    return [opening(model.LEADS[message.kind], message.when, now) for message in run.messages()]


def test_messages_are_the_windows_own_sentences_newest_first_and_the_rest_from_the_engine():
    """A warning or an error is a message, in the window's own words where its row has them, else "From the
    engine:" and the line as it stands, whatever its logger; two lines of information that say a training is not
    saved as it goes are messages too. A print never is. Messages that differ only in their numbers are one, with
    a count, said again as the newest; unusable driven recordings are one message whose sentence counts them, never
    summed; the newest 50 are kept, newest first, each opening with its kind and when it was said."""
    clock = _Clock(_at(11, 12, 10, 30))
    run = RunModel(_training(), clock=clock)
    for name, said in (
            ("patho_batch", "Failed simulations: 3 more in batch 6,150."),
            ("oom_split_simulation", "Batch 6,151 did not fit in the graphics card's memory and was run in smaller "
                                     "pieces."),
            ("oom_split_rows", "Batch 6,151 did not fit in the graphics card's memory and was run in smaller pieces."),
            ("save_put_off", "The save of the simulations at batch 6,150 was put off to the next save."),
            ("fit_save_failed", "The last save of the training failed (OSError: [Errno 28] No space left on device); "
                                "the kept training stands at epoch 40."),
            ("fit_save_failed_nothing_kept", "The last save of the training failed (OSError: [Errno 28] No space left "
                                             "on device); nothing of it is kept yet."),
            ("fit_folder_left", "A folder of a training that was not kept could not be removed; Records lists it as "
                                "a leftover to clear."),
            ("fit_resume_fallback", "The newest saved epoch of the kept training could not be read; it resumes from "
                                    "the epoch before it, epoch 39."),
            ("fit_no_summary", "This version of the training library gives no line per epoch, so the fit's epochs, "
                               "pace and loss are not shown."),
            ("fit_record_skipped", "An epoch's record could not be read from the training library (KeyError: "
                                   "'epoch_durations_sec'), so not every epoch is shown."),
            ("text_file_broken", "The run's text file could not be written (OSError: [Errno 28] No space left on "
                                 "device); the full text is still on the page."),
            ("chi_batch", "Unusable driven recordings were reported in 1 batch (latest: batch 6,150, 9,120 of "
                          "22,528).")):
        fresh = RunModel(_training(), clock=clock)
        _feed(fresh, _LINES[name], level="warning")
        assert [(m.kind, m.sentence, m.count) for m in fresh.messages()] == [("warn", said, 1)], name
    fresh = RunModel(_training(), clock=clock)
    _feed(fresh, _LINES["batch_failed"], level="error")
    assert [(m.kind, m.sentence) for m in fresh.messages()] == [
        ("err", "Batch 237 failed for lack of graphics-card memory and is being run again after a pause.")]
    for name, said in (
            ("fit_saving_off", "This training is not saved as it goes: a stop or a failure during its fit loses the "
                               "fit."),
            ("fit_not_kept", "Its simulation set is not kept (checkpointing is off), so this training is not saved as "
                             "it goes: a stop or a failure during its fit loses the fit.")):
        fresh = RunModel(_training(), clock=clock)
        _feed(fresh, _LINES[name])
        assert [(m.kind, m.sentence) for m in fresh.messages()] == [("info", said)], name
    for name in ("fit_listing_failed", "fit_mark_failed", "fit_stop_write_failed", "fit_finish_mark_failed",
                 "fit_remove_failed"):
        assert "pass" not in _LINES[name] and readings.sentence(readings.read(_LINES[name], "warning")) is None
        fresh = RunModel(_training(), clock=clock)
        _feed(fresh, _LINES[name], level="warning")
        assert [m.sentence for m in fresh.messages()] == [f"From the engine: {_LINES[name]}"], name

    rejection = ("library: rejection: Only 0.5% proposal samples were accepted. It may take a long time to collect "
                 "the remaining 1000 samples.")
    _feed(run, rejection, level="warning")
    _feed(run, "UserWarning: Only 3.2 GiB free where the training checkpoint needs about 10.3 GiB", level="warning")
    for level in ("info", "err"):
        _feed(run, _LINES["patho_batch"], rejection, "a line no row reads", level=level)
    assert [(m.kind, m.sentence) for m in run.messages()] == [
        ("warn", "From the engine: UserWarning: Only 3.2 GiB free where the training checkpoint needs about 10.3 GiB."),
        ("warn", f"From the engine: {rejection}")]
    clock.advance(60)
    _feed(run, rejection.replace("0.5%", "0.4%").replace("1000", "980"), level="warning")
    assert [(m.sentence, m.count, m.when) for m in run.messages()][0] == (
        f"From the engine: {rejection.replace('0.5%', '0.4%').replace('1000', '980')}", 2, clock())
    assert len(run.messages()) == 2, "the newest said again, at the top, counted"
    _feed(run, _LINES["chi_batch"], _LINES["chi_batch"].replace("6150/", "6152/").replace("9120/", "9000/"),
          level="warning")
    assert run.messages()[0].sentence == ("Unusable driven recordings were reported in 2 batches (latest: batch 6,152, "
                                          "9,000 of 22,528).")
    _feed(run, _LINES["batch_failed"], level="error")
    assert _openings(run, clock())[:2] == ["Error, Sun 11 Oct, 12:11.", "Attention, Sun 11 Oct, 12:11."]
    assert _openings(run, clock())[-1] == "Attention, Sun 11 Oct, 12:10."

    many = RunModel(_training(), clock=clock)
    words = [f"{'abcdefghijklmnopqrstuvwxyz'[n // 26]}{'abcdefghijklmnopqrstuvwxyz'[n % 26]}" for n in range(60)]
    _feed(many, *(f"warning {word}" for word in words), level="warning")
    assert [m.sentence for m in many.messages()] == [f"From the engine: warning {word}." for word in words[::-1][:50]]
    assert model.MESSAGES_KEPT == 50


def test_silence_is_said_in_minutes_and_needs_you_after_its_limit():
    """A run that says nothing, and whose solver does not move, is told: outside the fit "Attention" after 45 s and
    "needs you" after 10 minutes; in the fit it is information while the silence is shorter than twice the last
    epoch's time, and needs you from there, never while no epoch's time is known. Always in minutes or longer, never
    in seconds. The next line, row or step of the solver clears it, and an ended run is never silent."""
    clock = _Clock(_at(11, 12, 0, 0))
    run = RunModel(_training(), clock=clock)
    _simulating(run, clock)
    clock.advance(44)
    run.tick()
    assert run.messages() == () and run.needs_you() is None
    clock.advance(1)
    run.tick()
    assert [(m.kind, m.sentence, m.when) for m in run.messages()] == [("warn", "No output for 1 minute.", clock())]
    raised = clock()
    clock.advance(12 * 60 - 45 - 1)
    run.tick()
    assert [(m.sentence, m.when) for m in run.messages()] == [("No output for 12 minutes.", raised)]
    assert run.needs_you() == "No output for 12 minutes.", "past ten minutes"
    run.solver(1_000)
    run.solver(1_000)
    run.tick()
    assert run.messages()[0].sentence == "No output for 12 minutes.", "a solver count that did not move"
    run.solver(1_001)
    run.tick()
    assert run.messages() == () and run.needs_you() is None, "a step of the solver clears it"
    clock.advance(599)
    run.tick()
    assert run.needs_you() is None and run.messages()[0].sentence == "No output for 10 minutes."
    clock.advance(1)
    run.tick()
    assert run.needs_you() == "No output for 10 minutes."
    _feed(run, "a print")
    run.tick()
    assert run.messages() == () and run.needs_you() is None
    clock.advance(3 * 3600 + 25 * 60)
    run.tick()
    assert run.messages()[0].sentence == "No output for 3 hours 25 minutes."
    run.rows([_frame(10)])
    run.tick()
    assert run.messages() == ()

    fit = RunModel(_training(), clock=clock)
    _fitting(fit, clock)
    _feed(fit, _epoch(1, took=3780.0))
    clock.advance(12 * 60)
    fit.tick()
    assert [(m.kind, m.sentence) for m in fit.messages()] == [
        ("info", "No output for 12 minutes. This is normal during an epoch.")]
    assert fit.needs_you() is None
    clock.advance(2 * 3780 - 12 * 60 - 1)
    fit.tick()
    assert fit.needs_you() is None and fit.messages()[0].kind == "info"
    clock.advance(1)
    fit.tick()
    assert [(m.kind, m.sentence) for m in fit.messages()] == [
        ("warn", "No output for 2 hours 6 minutes, twice as long as the last epoch took.")]
    assert fit.needs_you() == "No output for 2 hours 6 minutes, twice as long as the last epoch took."
    waiting = RunModel(_training(), clock=clock)     # in the fit, before any epoch's time is known
    _fitting(waiting, clock)
    clock.advance(11 * 60)
    waiting.tick()
    assert [(m.kind, m.sentence) for m in waiting.messages()] == [
        ("info", "No output for 11 minutes. This is normal during an epoch.")]
    assert waiting.needs_you() is None, "never, while no epoch's time is known"
    for quiet in range(45, 4 * 3600, 37):
        assert "second" not in model._quiet_words(quiet), quiet
    fit.ended("stopped")
    fit.tick()
    assert fit.messages() == () and fit.needs_you() is None


def test_a_silence_under_an_hour_never_reads_sixty_minutes():
    """Under an hour a silence is said in whole minutes, 59 at the most, as a pace guards its own edge; from the
    hour on, as a length is written."""
    clock = _Clock(_at(11, 12, 0, 0))
    run = RunModel(_training(), clock=clock)
    _simulating(run, clock)
    heard = clock()
    for quiet, said in ((3569, "59 minutes"), (3570, "59 minutes"), (3599.9, "59 minutes"), (3600, "1 hour")):
        clock.set(heard + timedelta(seconds=quiet))
        run.tick()
        assert [m.sentence for m in run.messages()] == [f"No output for {said}."], quiet


def test_the_progress_panes_three_rules_live_on_as_functions():
    """The three rules of the old progress pane, as functions of rows read by their attributes: the row that drives
    is the one with the largest total among those that can measure, the solver's own left out, a tie going to the
    deeper row; the fit is found by the training library's counter row; a solver count that moved is a sign of
    life."""
    from core.gui import vt
    outer = vt.parse_bar(("err", 0), 0, "Training neural posterior:   0%|          | 0/1 [00:00<?, ?it/s]")
    rows = (outer, _frame(6_150, row=1), _frame(25_000, 50_000, desc="step (batch=2048)", row=2))
    assert rows[2].is_solver and not outer.informative
    assert model.driving_row(rows) is rows[1]
    assert model.driving_row((outer, rows[2])) is None
    tie = (_frame(3, 8, desc="a", row=0), _frame(5, 8, desc="b", row=1))
    assert model.driving_row(tie) is tie[1]
    counter = vt.parse_bar(("out", 0), 0, " Training neural network. Epochs trained: 62")
    assert model.fit_counter((outer, counter)) == 62 and model.fit_counter(rows) is None
    assert (model.moved(5, 6), model.moved(6, 6), model.moved(6, 5)) == (True, False, False)


def test_dates_numbers_and_durations_are_written_with_no_toolkit():
    """The date and the number are written as the look writes them, by the same functions, which live where no
    toolkit is loaded; a length of time is written in its two largest whole units, counted down."""
    from core.gui import design
    assert (design.date_text, design.number_text, design.MONTHS, design.WEEKDAYS) == (
        texts.date_text, texts.number_text, texts.MONTHS, texts.WEEKDAYS)
    at = datetime
    assert texts.date_text(at(2026, 10, 4, 12, 10, 59), at(2026, 10, 6, 9, 0)) == "Sun 4 Oct, 12:10"
    assert texts.date_text(at(2026, 10, 3, 14, 2)) == "3 Oct 2026, 14:02"
    assert texts.number_text(20_480_000) == "20,480,000" and texts.number_text(2048.0) == "2,048.0"
    for seconds, said in ((0, "less than a minute"), (59.9, "less than a minute"), (60, "1 minute"),
                          (12 * 60 + 59, "12 minutes"), (3600, "1 hour"), (72 * 60, "1 hour 12 minutes"),
                          (2 * 3600 + 30, "2 hours"), (86_400, "1 day"), (2 * 86_400 + 17 * 3600, "2 days 17 hours"),
                          (5 * 86_400 + 19 * 3600 + 59 * 60, "5 days 19 hours"), (-5, "less than a minute"),
                          (float("nan"), "less than a minute")):
        assert texts.duration_text(seconds) == said, seconds


# ── the run's controller: a run of the window, its files and its readers ──
# The names a page's run flag went by, which nothing under the window or the suites may name now: the run's state
# is the run controller's, and a test that wants a live run starts the stand-in run.
_FLAG_NAMES = ("_running", "_set_busy", "_active_cancel", "request_cancel_all", "RUN_STATE")
# The places that start a run under the window: the file, the class and the function each hands its dispatch.
_DISPATCHES = {("core/gui/panels/crossval_panel.py", "CrossValPanel", "compare"),
               ("core/gui/panels/crossval_panel.py", "CrossValPanel", "run_param_study_cli"),
               ("core/gui/panels/fdt_panel.py", "FdtPanel", "compare"),
               ("core/gui/panels/fdt_panel.py", "FdtPanel", "_run_fdt_guarded"),
               ("core/gui/panels/inference/infer_tab.py", "InferPanel", "orchestrator.simulated_inference"),
               ("core/gui/panels/inference/infer_tab.py", "InferPanel", "orchestrator.experimental_inference"),
               ("core/gui/panels/inference/posterior_tab.py", "PosteriorPanel", "orchestrator.build_posterior"),
               ("core/gui/panels/inference/prior_tab.py", "PriorPanel", "orchestrator.build_prior"),
               ("core/gui/panels/inference/tsnpe_tab.py", "TSNPEPanel", "orchestrator.tsnpe_round"),
               ("core/gui/panels/inference/validate_tab.py", "ValidatePanel", "orchestrator.validate_calibration"),
               ("core/gui/panels/reduction_panel.py", "ReductionPanel", "run_reduction_map"),
               ("core/gui/panels/simulate_panel.py", "SimulatePanel", "run_simulation_stream"),
               ("core/gui/panels/simulate_panel.py", "SimulatePanel", "export_animation")}


def _controller_on(tmp_path, clock):
    """The process's run controller, its clock and its folder of runs this test's own; the end of the test puts
    both back."""
    from core.gui.run.controller import controller
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    return held


def _entries(path) -> list:
    """A run's text file as ``(level, text)`` for each line written, a text's further lines joined back under it."""
    from core.gui.run.textfile import STAMP_WIDTH
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(" " * STAMP_WIDTH):
            level, text = out[-1]
            out[-1] = (level, f"{text}\n{line[STAMP_WIDTH:]}")
        else:
            _day, _time, level, text = line.split(" ", 3)
            out.append((level, text))
    return out


def test_a_run_goes_from_starting_to_running_to_ended_and_its_files_say_so(tmp_path, monkeypatch):
    """A training of the window, through a page's own dispatch and the stand-in run, from the start to its end.
    Its status goes from starting, at the start, to running at its first line, through its stages, and is
    published again at a save of the simulation set passed, to ended. Its history file is written at the start
    (running), at each change of state or stage, at the save passed and at an epoch, with what each says, and at
    the end; its text file holds the window's head (the run, and the memory limit in force with where it was
    set), every line of the run and how it ended, and the run's full text holds every line the file holds, in the
    file's order. The fit's pace, network and simulations in the history are what the next training of that size
    finds its pace by."""
    from core.artifacts import default_store
    from core.gui.run import history
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    monkeypatch.setenv("PRISM_VRAM_CEILING_GIB", "6")
    clock = Clock(_at(8, 19, 2, 11))
    held = _controller_on(tmp_path, clock)
    folder, run = tmp_path / "runs", "20261008T190211-training"
    text, stand_in = RunText(held), StandIn()

    def record():
        return history.read(folder / f"{run}.json")

    start_run(build_run_host(), stand_in, spec=_training())
    assert (held.status.state, held.status.run, held.status.place) == ("starting", run, ("inference", "training"))
    first = record()
    assert (first["outcome"], first["stage"], first["word"], first["name"], first["place"], first["memory_limit"],
            first["records_root"], first["started"], first["text"]) == (
        "running", None, "Training", "retrain_3", ["inference", "training"], 6.0, str(default_store().root),
        "2026-10-08T19:02:11-07:00", f"{run}.txt")
    head = [("window", f"Training retrain_3, run {run}."),
            ("window", "Memory limit: 6 GiB for each batch of simulations (set by PRISM_VRAM_CEILING_GIB).")]
    assert _entries(folder / f"{run}.txt") == head == text.lines

    def said(line):
        stand_in.say(line)
        turn_until(app, lambda: line in text.texts())

    clock.advance(5)
    said(_LINES["budget"])
    assert (held.status.state, held.status.stage, record()["stage"]) == ("running", None, None)
    said(_LINES["rotation_begins"])
    assert (held.status.stage, record()["stage"]) == ("Rotation", "Rotation")
    clock.advance(90)
    said(_LINES["simulation_writing"])
    assert (held.status.stage, record()["stage"], record()["simulation_set"]["folder"]) == (
        "Simulation", "Simulation", _FOLDER)
    published = len(text.statuses)
    clock.advance(500)
    stand_in.row(SIMULATION_ROW, 50, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 50)
    assert len(text.statuses) == published + 1 and text.statuses[-1].stage == "Simulation", "no status at the save"
    assert record()["simulation_set"]["at_end"] == 50, record()["simulation_set"]
    clock.advance(60)
    said(_epoch(1, took=60.0))
    fit = record()["fit"]
    assert (held.status.stage, record()["stage"], fit["epochs"], fit["best_epoch"], fit["seconds_per_epoch"],
            fit["network"], fit["simulations"]) == ("Fitting", "Fitting", 1, 1, 60.0, [256, 10], 20_480_000)
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert [(status.state, status.stage) for status in text.statuses] == [
        ("starting", None), ("running", None), ("running", "Rotation"), ("running", "Simulation"),
        ("running", "Simulation"), ("running", "Fitting"), ("ended", "Fitting")]
    assert [(status.state, status.outcome, status.run) for status in text.ended] == [("ended", "finished", run)]
    last = record()
    assert (last["outcome"], last["ended"], last["stage"]) == ("finished", "2026-10-08T19:13:06-07:00", "Fitting")
    written = _entries(folder / f"{run}.txt")
    assert written == text.lines, (written, text.lines)
    assert written[-1] == ("window", "The run finished.") and _LINES["budget"] in [t for _l, t in written]
    assert history.estimate(folder, word="Training", per="epoch", size={"network": (256, 10),
                                                                        "simulations": 20_480_000}) == (
        60.0, "the last training of this size, retrain_3, took about 1 minute per epoch")


def test_one_run_at_a_time_in_the_whole_program():
    """While a run is live no other starts anywhere in the process: a second page's dispatch starts nothing,
    returns False and says why among that page's own lines; the controller itself refuses one handed to it; and
    the live run goes on, its status the first run's."""
    from tests._fixtures import PaneCapture, build_run_host, qt_app
    from tests._runs import StandIn, start_run, turn_until
    from core.gui.run.controller import LIVE_TEXT, controller
    app = qt_app()
    first, other, stand_in = build_run_host(), build_run_host(), StandIn()
    pane, called = PaneCapture(other), []
    start_run(first, stand_in)
    try:
        assert other.dispatch(lambda: called.append("other")) is False
        assert controller().dispatch(lambda: called.append("handed"), spec=RunSpec("Run", None)) is False
        assert pane.lines == [("warning", LIVE_TEXT)], pane.lines
        assert controller().host() is first and first.is_running() and not other.is_running()
    finally:
        stand_in.release()
    turn_until(app, lambda: not controller().is_live())
    assert called == [], "a second run was started beside the first"


def test_a_refusal_before_anything_is_spent_is_no_run(tmp_path):
    """A refusal the stage makes before anything is spent -- the solver has taken no step, no stage after the
    run's first has begun, the store holds no record of a kind the run writes made since the start -- is no
    run, though it came after the run's first line: the controller is idle again with nothing to acknowledge,
    the run's two files are removed, so nothing of the refusal stays in a file, and the page shows the refusal
    in its yellow box, as it shows one made at the click. A record of a kind the run writes that was made
    before its start is not the run's: the store here holds one."""
    from core.artifacts import ArtifactStore, use_store
    from core.refusals import Refusal
    from core.gui.run.model import IDLE
    from tests._fixtures import asked, build_fdt_record, build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, stand_in = RunText(held), StandIn()
    with use_store(ArtifactStore(tmp_path / "Artifacts")) as store:
        build_fdt_record(store)                          # made before the run starts: nothing of the run's
        start_run(build_run_host(), stand_in, spec=_training(kinds=("posterior", "simulation", "fit", "fdt")))
        assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == [
            "20261008T190211-training.json", "20261008T190211-training.txt"]
        stand_in.say(_LINES["budget"])
        turn_until(app, lambda: held.status.state == "running")
        stand_in.fail(Refusal("The posterior name 'retrain_3' is taken.", field="name"))
        turn_until(app, lambda: not held.is_live())
    assert held.status == IDLE and not held.status.unacknowledged_end
    assert list((tmp_path / "runs").iterdir()) == [], "a refusal before anything was spent left a file"
    assert [(status.state, status.outcome) for status in text.ended] == [("idle", None)]
    box = asked()
    assert (box.kind, box.sentence) == ("refusal", "The posterior name 'retrain_3' is taken."), box


@pytest.mark.parametrize("spent", ["the solver stepped", "a later stage began", "a record was written"])
def test_a_refusal_after_something_was_spent_ends_the_run_as_failed(spent, tmp_path):
    """A refusal that comes once something is spent -- the solver's count moved, a stage after the run's first
    began, or a record of a kind the run writes was made since the start -- ends the run as failed: its files are
    kept, saying so, and the run's page says the refusal as the failure it now is, with no box, and with where to
    fix it, as the window says a refusal."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.fields import fix_sentence
    from core.gui.run import history
    from core.refusals import Refusal
    from tests._fixtures import all_asked, build_fdt_record, build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    stand_in, run, host = StandIn(), "20261008T190211-training", build_run_host()
    with use_store(ArtifactStore(tmp_path / "Artifacts")) as store:
        start_run(host, stand_in, spec=_training(kinds=("posterior", "simulation", "fit", "fdt")))
        if spent == "the solver stepped":
            stand_in.solver(5)
        elif spent == "a later stage began":
            stand_in.say(_LINES["rotation_begins"])
            stand_in.say(_LINES["simulation_writing"])
        else:
            build_fdt_record(store)
        stand_in.fail(Refusal("The read-back of the posterior was refused.", field="name"))
        turn_until(app, lambda: not held.is_live())
    assert (held.status.state, held.status.outcome, held.status.run) == ("ended", "failed", run), held.status
    kept = history.read(tmp_path / "runs" / f"{run}.json")
    assert (kept["outcome"], kept["failure"]["type"]) == ("failed", "Refusal"), kept
    assert (tmp_path / "runs" / f"{run}.txt").is_file()
    view = host.run_view
    assert all_asked() == [], "the refusal was told in a box"
    assert (view.failure.isVisibleTo(view), view.failure_message.text()) == (
        True, f"The read-back of the posterior was refused. {fix_sentence('name')}"), (
        "the refusal is not said on the run's page, or not with where to fix it")


def test_a_run_that_only_loads_shows_loading_and_leaves_nothing_to_acknowledge(tmp_path):
    """A run that only loads a record is published as loading from its start to its end, and ends straight into
    idle, with nothing to acknowledge."""
    from core.gui.run.model import IDLE
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, stand_in = RunText(held), StandIn(word="Prior load")
    start_run(build_run_host(), stand_in, spec=RunSpec("Prior load", ("inference", "prior"), stop_button="Stop loading",
                                                       loads_only=True, keeps_files=False, stop_asks=False))
    assert (held.status.state, held.status.loading, held.status.state_word) == ("starting", True, "loading")
    stand_in.say("the prior is read")
    turn_until(app, lambda: held.status.state == "running")
    assert (held.status.loading, held.status.state_word) == (True, "loading")
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert held.status == IDLE and not held.status.unacknowledged_end
    assert [status.loading for status in text.statuses] == [True, True, False], text.statuses
    assert [(status.state, status.outcome) for status in text.ended] == [("idle", "finished")]


def test_a_load_a_live_simulation_and_a_video_export_leave_no_file(tmp_path):
    """A prior's and a posterior's load, a live simulation and a video export, each with its own page's spec:
    after each, the folder of runs holds no history file and no text file, and the controller is idle with
    nothing to acknowledge."""
    from core import orchestrator
    from core.gui.panels.simulate_export import export_animation
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.panels.simulate_runner import run_simulation_stream
    from core.gui.run.model import IDLE
    from tests._fixtures import build_inference, build_part, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    folder, inference, live = tmp_path / "runs", build_inference(), build_part(SimulatePanel)
    cfg = SimpleNamespace(model="NADROWSKI", observation_mode="chi", reparam_rotate=True)
    for host, fn, args in ((inference.prior_panel, orchestrator.build_prior, (cfg, "p1", False)),
                           (inference.posterior_panel, orchestrator.build_posterior, (cfg, None, "q1", False)),
                           (live, run_simulation_stream, (cfg, 1.0, 10, 30.0)),
                           (live, export_animation, (None, "bundle.gif"))):
        spec = host.run_spec(fn, *args)
        stand_in = StandIn(word=spec.word)
        start_run(host, stand_in, spec=spec)
        stand_in.say("a line of the run")
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
        assert (sorted(folder.glob("*.json")), sorted(folder.glob("*.txt"))) == ([], []), spec.word
        assert held.status == IDLE and not held.status.unacknowledged_end, spec.word


def test_the_model_is_handed_the_pace_of_the_last_run_of_the_same_size(tmp_path, monkeypatch):
    """Before its first batches are timed, a training's sentence takes its pace, and its end, from the history:
    the newest ended run of the same word and the same simulations per batch in the folder of runs in use, with
    its source in words. A training that has timed its batches writes its own pace into its history, where the
    next run of that size finds it."""
    from core.gui.run import history
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    folder = tmp_path / "runs"
    history.write(folder, {"id": "20261001T090000-training", "word": "Training", "name": "retrain_aug",
                           "started": _at(1, 9, 0), "outcome": "finished",
                           "simulation_set": {"folder": "other", "per_batch": 2048, "seconds_per_batch": 9.7}})
    answers, real = [], history.estimate

    def estimate(where, **asked):
        said = real(where, **asked)
        answers.append((where, asked["word"], asked["per"], said))
        return said

    monkeypatch.setattr(history, "estimate", estimate)
    clock = Clock(_at(8, 19, 2, 11))
    held, stand_in = _controller_on(tmp_path, clock), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    stand_in.say(_LINES["budget"])
    stand_in.say(_LINES["simulation_writing"])
    stand_in.row(SIMULATION_ROW, 5, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 5)
    now = clock()
    assert held.model.sentence() == (
        f"Simulating: batch 5 of 10,000, 9.7 s per batch; the simulations should be done by "
        f"{texts.date_text(now + timedelta(seconds=9_995 * 9.7), now)}.")
    assert (folder, "Training", "batch",
            (9.7, "the last training of this size, retrain_aug, took 9.7 s per batch")) in answers, answers
    clock.advance(80)
    stand_in.row(SIMULATION_ROW, 25, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 25)
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    kept = history.read(folder / "20261008T190211-training.json")["simulation_set"]
    assert (kept["per_batch"], kept["seconds_per_batch"]) == (2048, 4.0), kept
    assert real(folder, word="Training", per="batch", size={"per_batch": 2048}) == (
        4.0, "the last training of this size, retrain_3, took 4.0 s per batch")


def test_the_fits_expected_epochs_are_read_once_from_the_newest_finished_training_of_its_kind(tmp_path):
    """A training's expected epochs are read once, at its start, from the store's posteriors: the newest
    finished one of the same model, kind of observation and narrowing, whose record holds its epochs, with its
    label in the words; one that is not finished, one of a narrowing round, one of another kind of observation,
    of another model, or with no epochs is passed over, and so is an older one of the kind. The store's
    posteriors are listed once for each run; the count reads those lists alone, since another part of the
    window may read the store in use while a run ends."""
    import json
    from core.artifacts import use_store
    from core.artifacts.store import MANIFEST, Summary
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    root = tmp_path / "Artifacts"

    def posterior(name, mode, day, epochs, model="NADROWSKI", complete=True, amortized=True):
        path = root / "posteriors" / f"{name}__x"
        path.mkdir(parents=True)
        (path / MANIFEST).write_text(json.dumps({"config": {"model": model}, "body": {
            "mode": mode, "training": {"epochs_trained": epochs}}}), encoding="utf-8")
        return Summary("posterior", name, name, _at(day, 10, 0).isoformat(), "", path, complete, None, mode=mode,
                       amortized=amortized)

    listed = [posterior("unfinished", "chi", 7, 70, complete=False),
              posterior("narrowed", "chi", 7, 60, amortized=False), posterior("no_epochs", "chi", 7, None),
              posterior("other_model", "chi", 6, 50, model="HOPF"), posterior("forced_one", "forced", 6, 90),
              posterior("retrain_aug", "chi", 5, 130), posterior("older", "chi", 1, 77)]
    calls = []
    store = SimpleNamespace(root=root, list=lambda kind: (calls.append(kind), listed if kind == "posterior" else [])[1])
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    with use_store(store):
        for turn in (1, 2):
            stand_in = StandIn()
            start_run(build_run_host(), stand_in, spec=_training())
            stand_in.say(_LINES["budget"])
            stand_in.say(_epoch(1, took=60.0))
            turn_until(app, lambda: held.model.stage == "Fitting")
            assert held.model.axis().follows == (
                "About 130 epochs: the last training of this kind, retrain_aug, ran 130."), held.model.axis()
            stand_in.release()
            turn_until(app, lambda: not held.is_live())
            assert [kind for kind in calls if kind == "posterior"] == ["posterior"] * turn, calls


def test_a_continued_runs_failed_simulations_count_its_earlier_runs(tmp_path):
    """A training that continues a simulation set counts its failed simulations over every run of that set: the
    earlier runs' counts are read from their history files once the set's folder is named, and the health table
    says "over 3 runs". The run writes its own counts into its history for the next."""
    from core.gui.run import history
    from core.gui.run.model import HealthRow
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    folder = tmp_path / "runs"
    for name, day, failed in (("a", 1, 100), ("b", 3, 50)):
        history.write(folder, {"id": f"2026100{day}T090000-training", "word": "Training", "started": _at(day, 9, 0),
                               "outcome": "failed", "simulation_set": {"folder": _FOLDER},
                               "health": {"failed": failed, "simulated": 10_000}})
    held, stand_in = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    stand_in.say(_LINES["simulation_resuming"])
    stand_in.say(_LINES["patho_total"])
    turn_until(app, lambda: held.model.failed() is not None)
    assert held.model.health()[0] == HealthRow("Failed simulations", "0.15 % over 3 runs", "Fine")
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    kept = history.read(folder / "20261008T190211-training.json")
    assert (kept["health"], kept["simulation_set"]["at_start"]) == (
        {"failed": 30_720, "simulated": 20_480_000}, 236), kept


def test_every_place_that_starts_a_run_hands_the_controller_a_spec():
    """Every call that starts a run under the window is a page's own ``self.dispatch`` (the base panel's one
    forward to the controller, and the controller's own start of a run's call again from its page's offer, with the
    spec that page made, aside), each in a class whose ``run_spec`` answers its function (a call outside every
    class answers to none), read off the files' syntax trees; and the spec of each kind of run, read off
    its own page's ``run_spec`` with arguments of the call's shape, holds the run's word, its place, its stop
    button, the kinds of record it writes, whether it only loads, whether it keeps files, whether it is tried
    again and whether its stop asks, with a training's stages, patience, epoch limit and facts. The stand-in
    run's places are the window's own."""
    import ast
    from core import config, orchestrator
    from core.FDT.compare import compare
    from core.FDT.cross_validation import run_param_study_cli
    from core.Reduction.sweep import run_reduction_map
    from core.gui.panels import fdt_panel
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.simulate_export import export_animation
    from core.gui.panels.simulate_runner import run_simulation_stream
    from tests import _runs
    from tests._fixtures import build_window, qt_app
    qt_app()
    found, forwards = set(), []
    for path in sorted((REPO / "core" / "gui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = {id(node): node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "dispatch"}
        in_a_class = set()
        for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            answers = any(isinstance(item, ast.FunctionDef) and item.name == "run_spec" for item in cls.body)
            for call in [node for node in ast.walk(cls) if id(node) in calls]:
                in_a_class.add(id(call))
                where = (path.relative_to(REPO).as_posix(), cls.name, ast.unparse(call.args[0]))
                if ast.unparse(call.func) == "self.dispatch" and answers:
                    found.add(where)
                else:
                    forwards.append(where)
        forwards += [(path.relative_to(REPO).as_posix(), None, ast.unparse(call.args[0]))
                     for key, call in calls.items() if key not in in_a_class]     # outside every class: no run_spec
    assert found == _DISPATCHES, sorted(found ^ _DISPATCHES)
    assert forwards == [("core/gui/panels/base_panel.py", "BasePanel", "fn"),
                        ("core/gui/run/controller.py", "RunController", "spec.fn")], forwards

    window = build_window()
    inference, temperature = window.part("inference"), window.part("temperature")
    reduction, live = window.part("reduction"), window.part("live")
    one_cell, sweep = temperature.panels()
    cfg = SimpleNamespace(model="NADROWSKI", observation_mode="chi", reparam_rotate=True)
    writer = SimpleNamespace(id="f1", name="cell_7")
    writers = {"s": SimpleNamespace(id="s1", name="study-s"), "temp": SimpleNamespace(id="t1", name="study-temp")}
    yes, no = True, False
    kinds_of = {
        "prior built": (inference.prior_panel, orchestrator.build_prior, (cfg, None, True), {},
                        ("Prior build", ("inference", "prior"), "Stop the prior build…", ("prior",), no, yes, no,
                         yes)),
        "prior loaded": (inference.prior_panel, orchestrator.build_prior, (cfg, "p1", False), {},
                         ("Prior load", ("inference", "prior"), "Stop loading", (), yes, no, no, no)),
        "training": (inference.posterior_panel, orchestrator.build_posterior, (cfg, None, None, True), {},
                     ("Training", ("inference", "training"), "Stop training…", ("posterior", "simulation", "fit"), no,
                      yes, yes, yes)),
        "posterior loaded": (inference.posterior_panel, orchestrator.build_posterior, (cfg, None, "q1", False), {},
                             ("Posterior load", ("inference", "training"), "Stop loading", (), yes, no, no, no)),
        "calibration": (inference.validate_panel, orchestrator.validate_calibration, (cfg, None, None), {},
                        ("Check", ("inference", "checks"), "Stop the calibration…", ("calibration",), no, yes, no,
                         yes)),
        "inference, simulated": (inference.infer_panel, orchestrator.simulated_inference, (cfg, None, 1.5), {},
                                 ("Inference", ("inference", "infer"), "Stop the inference…",
                                  ("observation", "inference"), no, yes, no, yes)),
        "inference, on recordings": (inference.infer_panel, orchestrator.experimental_inference, (cfg, None, None),
                                     {}, ("Inference", ("inference", "infer"), "Stop the inference…",
                                          ("observation", "inference"), no, yes, no, yes)),
        "narrowing round": (inference.tsnpe_panel, orchestrator.tsnpe_round, (cfg, None, None, None), {},
                            ("Narrowing round", ("inference", "narrowing"), "Stop the narrowing round…",
                             ("posterior", "simulation", "fit"), no, yes, yes, yes)),
        "one cell": (one_cell, fdt_panel._run_fdt_guarded, (cfg,), {"writer": writer},
                     ("Measurement", ("temperature", "one_cell"), "Stop measurement…", ("fdt",), no, yes, no, yes)),
        "sweep study": (sweep, run_param_study_cli, (cfg,), {"writers": writers},
                        ("Sweep study", ("temperature", "sweep"), "Stop sweep study…", ("fdt",), no, yes, no, yes)),
        "comparison, one cell": (one_cell, compare, ("cells", ["f1"]), {"name": "both"},
                                 ("Comparison", ("temperature", "one_cell"), "Stop the comparison…", ("fdt",), no,
                                  yes, no, yes)),
        "comparison, sweeps": (sweep, compare, ("sweeps", ["s1"]), {"name": "both"},
                               ("Comparison", ("temperature", "sweep"), "Stop the comparison…", ("fdt",), no, yes,
                                no, yes)),
        "reduction map": (reduction, run_reduction_map, (cfg,), {},
                          ("Reduction map", ("reduction", None), "Stop computing…", (), no, yes, no, yes)),
        "live simulation": (live, run_simulation_stream, (cfg, 1.0, 10, 30.0), {},
                            ("Live simulation", ("live", None), "Stop simulation", (), no, no, no, no)),
        "video": (live, export_animation, (None, "bundle.gif"), {},
                  ("Video", ("live", None), "Stop saving…", (), no, no, no, yes)),
    }
    specs = {}
    for kind, (panel, fn, args, kwargs, row) in kinds_of.items():
        spec = specs[kind] = panel.run_spec(fn, *args, **kwargs)
        assert (spec.word, spec.place, spec.stop_button, spec.kinds, spec.loads_only, spec.keeps_files, spec.retries,
                spec.stop_asks) == row, kind
        assert spec.expected_epochs is None and spec.fn is None, kind
        trains = kind in ("training", "narrowing round")
        assert (spec.simulation_set, spec.stages == _TRAINING) == (trains, trains), kind
        if not trains and kind not in ("one cell", "sweep study"):
            assert (spec.stages, spec.facts) == ((), {}), kind
    assert specs["calibration"].name == "Calibration" and specs["one cell"].name == "cell_7"
    assert (specs["sweep study"].name, specs["comparison, sweeps"].name) == ("study", "both")
    assert (specs["one cell"].records, specs["sweep study"].records) == (("f1",), ("s1", "t1"))
    assert specs["live simulation"].own_view and not specs["video"].own_view
    assert [stage.name for stage in specs["one cell"].stages] == ["Checks", "Spontaneous", "Driven"]
    assert specs["sweep study"].stages == (Stage("S sweep", ("sweep_phase",), "sweep_point_b"),
                                           Stage("T_a/T sweep", ("sweep_phase",), "sweep_point_b"))
    for kind, narrowed in (("training", False), ("narrowing round", True)):
        spec = specs[kind]
        assert (spec.patience, spec.epoch_limit) == (config.TRAINING_STOP_AFTER_EPOCHS, config.TRAINING_MAX_NUM_EPOCHS)
        assert spec.facts == {"model": "NADROWSKI", "observation": "chi", "narrowed": narrowed,
                              "network": (config.NSF_HIDDEN_FEATURES, config.NSF_NUM_TRANSFORMS), "rotated": True,
                              "cadence": orchestrator.TRAINING_CHECKPOINT_EVERY}, kind
    given = inference.posterior_panel.run_spec(orchestrator.build_posterior, cfg, None, None, True,
                                               stop_after_epochs=7, max_num_epochs=40, hidden_features=64,
                                               num_transforms=3, checkpoint_every=25)
    assert (given.patience, given.epoch_limit, given.facts["network"], given.facts["cadence"]) == (7, 40, (64, 3), 25)
    with pytest.raises(NotImplementedError, match="has no run spec for"):
        sweep.run_spec(run_reduction_map, cfg)
    assert isinstance(sweep, CrossValPanel)
    held = {type(panel).__name__: window._places[panel] for panel in window._places}
    assert held == _runs.PLACES, (held, _runs.PLACES)


def test_no_test_and_no_part_of_the_window_reaches_a_run_through_a_panels_flag():
    """Nothing under the program or the suites names a page's old run flag -- a name, an attribute, a name
    defined or imported, a keyword, or a string that is one of them -- read off each file's syntax tree, its
    docstrings and comments aside, this test's own names excepted: whether a run is live, and its stop, are the
    run controller's, and a test that wants a live run starts the stand-in run."""
    import ast
    snippet = ast.parse("\n".join([
        "BasePanel._running = True",                                     # 1: an attribute
        "from core.gui.panels.base_panel import RUN_STATE as state",     # 2: a name imported
        "monkeypatch.setattr(BasePanel, '_set_busy', None)",             # 3: a string that is one
        "def request_cancel_all(_active_cancel=None): pass",             # 4: a name defined, a parameter
        "f(_active_cancel=1)",                                           # 5: a keyword
        "\"\"\"Says _running in its words, which is not code.\"\"\"",   #    a sentence: not read
        "is_running = controller().is_live()"]))                         #    another name
    assert [(line, name) for line, name in _flag_names(snippet)] == [
        (1, "_running"), (2, "RUN_STATE"), (3, "_set_busy"), (4, "request_cancel_all"), (4, "_active_cancel"),
        (5, "_active_cancel")], _flag_names(snippet)
    found = []
    for root in ("core", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            for line, name in _flag_names(ast.parse(path.read_text(encoding="utf-8"))):
                found.append(f"{path.relative_to(REPO).as_posix()}:{line} {name}")
    assert found == [], found


def _flag_names(tree) -> list:
    """Each place in ``tree`` that names a page's old run flag, as ``(line, name)`` in the order of the lines: a
    name, an attribute, a name defined or imported, a parameter, a keyword, or a string that is exactly one of
    them. The scan of the program and the suites leaves out this module's list of the names and the test that
    holds the scanner's own example."""
    import ast
    own = set()
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "_FLAG_NAMES"
                                                  for target in node.targets))
                or (isinstance(node, ast.FunctionDef)
                    and node.name == "test_no_test_and_no_part_of_the_window_reaches_a_run_through_a_panels_flag")):
            own |= {id(inner) for inner in ast.walk(node)}
    found = []
    for node in ast.walk(tree):
        if id(node) in own:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            own.add(id(node.value))                   # a docstring or a lone string: words, not code
            continue
        said = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                else node.arg if isinstance(node, (ast.arg, ast.keyword))
                else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None)
        names = ({said} if said else set()) | ({node.name.rsplit(".", 1)[-1], node.asname}
                                                if isinstance(node, ast.alias) else set())
        found += [(node.lineno, name) for name in sorted(names & set(_FLAG_NAMES)) if id(node) not in own]
    return sorted(found, key=lambda place: place[0])


def test_the_run_state_reaches_a_window_through_a_method_of_its_own_and_goes_with_it():
    """The process's one run controller outlives every window; each window, and each page that hears a run's
    end, connects to it through a method of its own object, so a window deleted leaves the controller exactly
    the connections it had before the window was built."""
    from PySide6.QtCore import QCoreApplication, QEvent, SIGNAL
    from core.gui.run.controller import controller
    from tests._fixtures import build_window, qt_app
    qt_app()
    held = controller()

    def counts():
        return (held.receivers(SIGNAL("status_changed(PyObject)")), held.receivers(SIGNAL("run_ended(PyObject)")))

    before = counts()
    window = build_window()
    window.part("temperature")
    assert counts() == (before[0] + 1, before[1] + 2), (before, counts())
    window.deleteLater()
    QCoreApplication.sendPostedEvents(window, QEvent.DeferredDelete)
    assert counts() == before, (before, counts())


def test_a_part_built_during_a_run_asks_the_controller_and_comes_up_locked():
    """A part, and a whole window, built while a run is live -- here the run of a host no window holds --
    ask the process's run controller: the part comes up locked, and the window built then says at its foot
    that a run is live. The run's end frees them, and the foot then says how it ended."""
    from core.gui.panels.reduction_panel import ReductionPanel
    from tests._fixtures import build_run_host, build_window, qt_app
    from tests._runs import running
    qt_app()
    with running(build_run_host()):
        window = build_window()
        panel = window.panel(ReductionPanel)
        assert not panel.controls.isEnabled(), "a part built during a run is not locked"
        assert window.shell.rail.foot._title == "Training is running", window.shell.rail.foot._title
    assert panel.controls.isEnabled() and window.shell.rail.foot._title == "Training finished"


def test_the_runs_text_file_reaches_the_disk_on_a_thread_of_its_own(tmp_path, monkeypatch):
    """At most every thirty seconds of the run's clock, the controller's look at the run asks the disk to hold
    the run's text file, on a thread of its own: the window's thread does not wait for the disk, which the
    engine may be writing gigabytes to."""
    import os
    import threading
    from core.gui.run import textfile
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    asked, entered, go = [], threading.Event(), threading.Event()

    def fsync(handle):
        asked.append(threading.current_thread() is threading.main_thread())
        entered.set()
        go.wait(2)

    monkeypatch.setattr(textfile, "os", SimpleNamespace(fsync=fsync, dup=os.dup, close=os.close))
    clock = Clock(_at(8, 19, 2, 11))
    held, stand_in = _controller_on(tmp_path, clock), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    held.tick()
    assert entered.wait(2) and asked == [False] and not go.is_set(), "the window's thread waited for the disk"
    go.set()
    clock.advance(29)
    held.tick()
    clock.advance(1)
    held.tick()
    turn_until(app, lambda: len(asked) == 2)
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert asked == [False, False, True], "the end's request is the window's, once: " + repr(asked)


def test_a_text_file_that_breaks_at_its_head_is_said_and_the_run_goes_on(tmp_path, monkeypatch):
    """A run whose text file cannot be written from its first line says so, as a line that breaks the file is
    said: among the run's full text at warning, and as a message of its page, in the window's own words. The
    run goes on, and ends as it ends."""
    from core.gui.run import textfile
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()

    def refused(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(textfile, "open", refused, raising=False)
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, stand_in = RunText(held), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    sentence = _LINES["text_file_broken"]
    assert ("warning", sentence) in text.lines, text.lines
    assert held.model.messages()[0].sentence == sentence
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert (held.status.state, held.status.outcome) == ("ended", "finished")


def test_the_head_says_the_memory_limit_in_force_and_where_it_was_set(tmp_path, monkeypatch):
    """The run's head names the memory limit in force for each batch of simulations, read once at its start, and
    where it was set: by the variable of the environment, which wins, or in step 1 of the window, which a variable
    that is not a number leaves in force; or says that none is set. The limit is the page's fact the health table
    reads."""
    from core import config
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    for variable, setting, said, limit in (
            ("6", 2.5, "Memory limit: 6 GiB for each batch of simulations (set by PRISM_VRAM_CEILING_GIB).", 6.0),
            (None, 2.5, "Memory limit: 2.5 GiB for each batch of simulations (set in step 1, Model and options).",
             2.5),
            ("abc", 2.5, "Memory limit: 2.5 GiB for each batch of simulations (set in step 1, Model and options).",
             2.5),
            (None, 0.0, "No memory limit is set on this computer.", None)):
        if variable is None:
            monkeypatch.delenv("PRISM_VRAM_CEILING_GIB", raising=False)
        else:
            monkeypatch.setenv("PRISM_VRAM_CEILING_GIB", variable)
        monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", setting)
        text, stand_in = RunText(held), StandIn()
        start_run(build_run_host(), stand_in, spec=_training())
        assert text.lines[1] == ("window", said), text.lines
        assert held.model.spec.facts["memory_limit"] == limit
        stand_in.release()
        turn_until(app, lambda: not held.is_live())


def test_the_controllers_look_hands_the_model_the_solvers_count_and_the_time(tmp_path):
    """Each second the controller looks at a live run and hands its model the solver's count and the time. A
    solver that stepped is a sign of life, so a run that prints nothing while its solver steps is not silent; a
    run that does neither is, and is said to be from its first minute."""
    from core.progress import SOLVER
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, stand_in = _controller_on(tmp_path, clock), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    held.tick()
    clock.advance(60)
    before = SOLVER.steps
    stand_in.solver(5)
    turn_until(app, lambda: SOLVER.steps != before)
    held.tick()
    assert held.model.messages() == (), held.model.messages()
    clock.advance(60)
    held.tick()
    assert [(message.kind, message.sentence) for message in held.model.messages()] == [
        ("warn", "No output for 1 minute.")], held.model.messages()
    stand_in.release()
    turn_until(app, lambda: not held.is_live())


def test_an_end_acknowledged_is_written_and_a_runs_place_is_found_by_its_id(tmp_path):
    """Someone has seen how a run ended: acknowledged by its id, the run the controller holds leaves the
    controller idle and its history file says so, and an older run's file, acknowledged by its id, says so too.
    A run's place is found by its id: the held run's from its spec, an older run's from its file, and none for an
    id no run has."""
    from core.gui.run import history
    from core.gui.run.model import IDLE
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    folder, older, run = tmp_path / "runs", "20261001T090000-measurement", "20261008T190211-training"
    history.write(folder, {"id": older, "word": "Measurement", "started": _at(1, 9, 0), "ended": _at(1, 9, 30),
                           "outcome": "finished", "place": ["temperature", "one_cell"]})
    held, stand_in = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), StandIn()
    start_run(build_run_host(), stand_in, spec=_training())
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert (held.status.unacknowledged_end, held.status.run) == (True, run), held.status
    assert (held.where(run), held.where(older), held.where("20261002T090000-video")) == (
        ("inference", "training"), ("temperature", "one_cell"), None)
    held.acknowledge(run)
    assert held.status == IDLE, held.status
    assert history.read(folder / f"{run}.json")["acknowledged"] is True, "the end acknowledged is not in its file"
    held.acknowledge(older)
    assert history.read(folder / f"{older}.json")["acknowledged"] is True, "the older run's file says nothing"


@pytest.mark.parametrize("fails", ["at the start", "during the run", "at the end", "every time"])
def test_a_history_file_that_cannot_be_written_is_said_once_and_tried_again(fails, tmp_path, monkeypatch):
    """A write of the run's history file that fails stops nothing: the run goes on and ends as it ends, and no
    panel stays locked. The failure is said once, in the window's words, in the run's full text and in its text
    file, after the file's head, wherever the write failed, the run's first moment among them; the full text
    holds the file's lines in the file's order. The history is tried again at the run's next moment that writes
    it: a write that failed once leaves the run's end in its history, and the end's own failure is mended when
    the end is acknowledged; a file that can never be written is said once all the same."""
    from core.gui.run import history
    from core.gui.run.controller import HISTORY_BROKEN_TEXT
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    real, calls = history.write, []
    error = OSError(13, "Permission denied")
    failing = {"at the start": (1,), "during the run": (2,), "at the end": (3,), "every time": None}[fails]

    def write(folder, record):
        calls.append(record.get("outcome"))
        if failing is None or len(calls) in failing:
            raise error
        return real(folder, record)

    monkeypatch.setattr(history, "write", write)
    folder, run = tmp_path / "runs", "20261008T190211-training"
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, stand_in, host = RunText(held), StandIn(), build_run_host()
    start_run(host, stand_in, spec=_training())
    stand_in.say(_LINES["rotation_begins"])
    turn_until(app, lambda: held.status.stage == "Rotation")
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    said = ("window", HISTORY_BROKEN_TEXT.format(f"{type(error).__name__}: {error}"))
    assert (held.status.state, held.status.outcome, len(text.ended)) == ("ended", "finished", 1), held.status
    assert host.controls.isEnabled(), "a broken history left the panels locked"
    assert calls == ["running", "running", "finished"], calls
    written = _entries(folder / f"{run}.txt")
    assert (text.lines.count(said), written.count(said)) == (1, 1), (text.lines, written)
    assert written == text.lines, (written, text.lines)
    kept = history.read(folder / f"{run}.json") or {}
    assert kept.get("outcome") == {"at the end": "running", "every time": None}.get(fails, "finished"), kept
    held.acknowledge(run)
    kept = history.read(folder / f"{run}.json") or {}
    assert (kept.get("outcome"), kept.get("acknowledged")) == (
        (None, None) if fails == "every time" else ("finished", True)), kept


def test_a_stop_over_a_failure_and_a_stop_too_late_leave_the_full_text_in_the_files_order(tmp_path):
    """What the worker says itself once the run's function has ended -- the failure a stop may have replaced,
    with its traceback, and a stop that came too late to stop anything -- reaches the run's full text where the
    run's text file holds it: before how the run ended. The full text holds the file's lines in the file's
    order."""
    from core.gui.streams import WorkerCancelled
    from core.gui.worker import CANCEL_TOO_LATE_TEXT, IN_FLIGHT_TEXT
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))

    def stopped_over_a_failure():
        try:
            raise RuntimeError("a failure the run may already have dealt with")
        except RuntimeError:
            raise WorkerCancelled()

    def finished_with_a_stop_too_late():
        try:
            raise RuntimeError("something the run recovered from")
        except RuntimeError:
            held.request_stop()                       # inside a handler: deferred, and nothing checks again
            print("the last thing the run writes")
        return "the payload"

    for fn, outcome, note in ((stopped_over_a_failure, "stopped", IN_FLIGHT_TEXT),
                              (finished_with_a_stop_too_late, "finished", CANCEL_TOO_LATE_TEXT)):
        text = RunText(held)
        start_run(build_run_host(), fn, spec=_training())
        turn_until(app, lambda: not held.is_live())
        written = _entries(tmp_path / "runs" / f"{held.status.run}.txt")
        assert held.status.outcome == outcome, held.status
        assert ("warning", note) in written[:-1], written
        assert written == text.lines, (written, text.lines)


def test_a_start_that_fails_before_its_worker_is_queued_leaves_nothing_locked_and_no_file(tmp_path, monkeypatch):
    """A start that raises once every panel is locked and before its worker is queued -- here a page hands its
    run a folder to watch of a shape no folder has -- raises to the page as it came, and is taken back: no run is
    live, the status is published idle again, every panel is free, the page's results are given back from the
    run's page, and the run's two files are removed. A load taken back so, which never took the column, leaves a
    stopped run's page where it was."""
    from core.gui.panels.base_panel import BasePanel
    from core.gui.run.model import IDLE
    from tests._fixtures import build_run_host, choose, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, host, other = RunText(held), build_run_host(), build_run_host()
    with pytest.raises(TypeError):
        start_run(host, StandIn(), spec=_training(), watch_dir=5)
    assert (held.is_live(), held.status) == (False, IDLE), held.status
    assert [status.state for status in text.statuses] == ["starting", "idle"], text.statuses
    assert all(panel.controls.isEnabled() for panel in BasePanel._instances) and other.controls.isEnabled()
    assert (host.results.currentWidget() is host.figures, host.run_view.isAncestorOf(host.figure_stack)) == (
        True, False), "the run's page kept the page's results for a run that never started"
    assert list((tmp_path / "runs").iterdir()) == [], "a start taken back left a file"
    start_run(host, StandIn(), spec=_training())
    choose(monkeypatch, words="Stop training")
    host.run_view.stop_button.click()
    turn_until(app, lambda: not held.is_live())
    with pytest.raises(TypeError):
        start_run(host, StandIn(word="Prior load"), spec=_load(), watch_dir=5)
    assert (host.results.currentWidget() is host.run_view, host.run_view.mode, host.run_view.sentence.text()) == (
        True, "ended", "Training was stopped on Thu 8 Oct, 19:02, as it started. Nothing was simulated."), (
        "a load taken back lost the stopped run's page")


@pytest.mark.parametrize("where", ["at its end", "as it is handed the run", "at its first drawing"])
def test_a_run_page_that_cannot_be_drawn_holds_up_neither_the_runs_start_nor_its_end(where, tmp_path,
                                                                                    monkeypatch):
    """The run's page is drawn inside the run's own sequence and never in its way. A page that cannot be drawn
    as its run ends leaves that end whole -- every panel free, the status published, the end said once, the run's
    payload let go -- and the failure goes to the program's last-resort error. One that cannot be drawn as its run
    starts, as it is handed the run or at its first drawing, fails the start, which is taken back like any other:
    no run live, the status published idle, every panel free, the page's results given back, no file kept."""
    from core.gui.panels.base_panel import BasePanel
    from core.gui.run.model import IDLE
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    text, host, other = RunText(held), build_run_host(), build_run_host()
    view, hooked = host.run_view, []
    monkeypatch.setattr("sys.excepthook", lambda kind, error, trace: hooked.append(str(error)))
    drawn = view.show_status

    def broken(*_args, **_kwargs):
        raise RuntimeError("the run's page cannot be drawn")

    if where == "at its end":
        monkeypatch.setattr(view, "show_status",
                            lambda status, model: drawn(status, model) if status.live else broken())
        stand_in = StandIn()
        start_run(host, stand_in, spec=_training())
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
        app.processEvents()
        run = held._run
        assert ([status.state for status in text.statuses], [status.outcome for status in text.ended]) == (
            ["starting", "ended"], ["finished"]), "a page that could not be drawn held up the run's end"
        assert (run.worker, run.token, held.status.state) == (None, None, "ended"), "the run's payload was kept"
        assert hooked == ["the run's page cannot be drawn"], hooked
    else:
        monkeypatch.setattr(view, "begin" if where == "as it is handed the run" else "show_status", broken)
        with pytest.raises(RuntimeError, match="the run's page cannot be drawn"):
            start_run(host, StandIn(), spec=_training())
        assert (held.is_live(), held.status) == (False, IDLE), held.status
        assert [status.state for status in text.statuses][-1:] == ["idle"], text.statuses
        assert host.results.currentWidget() is host.figures, "the run's page kept the page's results"
        assert list((tmp_path / "runs").glob("*")) == [], "a start taken back left a file"
    assert all(panel.controls.isEnabled() for panel in BasePanel._instances) and other.controls.isEnabled()


# ── this computer: kept awake while a run is live, and the reason Windows is given not to end the session ──
# What the calls to Windows are named, which no file under core/ but the one that makes them names.
_WINDOWS_CALLS = ("SetThreadExecutionState", "ShutdownBlockReason", ".alert(")
# The reason a training gives at each stage, and one whose fit is not saved as it goes.
_RUNNING = ("Training retrain_3 is running. Shutting down stops it. Nothing has been simulated yet. The rotation will "
            "be computed again next time.")
_SIMULATING = "Training retrain_3 is simulating. Shutting down stops it; the simulations done so far are saved first."
_FITTING = ("Training retrain_3 is fitting. Shutting down loses the epoch in progress; the epochs finished so far are "
            "kept.")
_SAVING = ("Training retrain_3 is saving its posterior. Shutting down stops it; the finished training is kept, and "
           "resuming it writes the posterior.")
_LOSES_THE_FIT = "Shutting down loses the fit: nothing of it can be saved. The simulations are kept."
# Every other kind of run: its spec, as its page hands it, and the reason it gives.
_OTHER_REASONS = (
    (RunSpec("Prior build", ("inference", "prior"), stop_button="Stop the prior build…", kinds=("prior",)),
     "Prior build is running. Shutting down stops it. Nothing of the prior is kept; it is built again from the start "
     "next time."),
    (RunSpec("Check", ("inference", "checks"), name="Calibration", stop_button="Stop the calibration…"),
     "Check Calibration is running. Shutting down stops it. A check writes its record only when it finishes, so "
     "nothing of this one is kept."),
    (RunSpec("Inference", ("inference", "infer"), stop_button="Stop the inference…"),
     "Inference is running. Shutting down stops it. The inference is not kept. An observation it has already written "
     "stays in Records."),
    (RunSpec("Measurement", ("temperature", "one_cell"), name="cell_a_3oct", stop_button="Stop measurement…"),
     "Measurement cell_a_3oct is running. Shutting down stops it. The record cell_a_3oct is kept, marked unfinished, "
     "with what it has measured so far. A measurement cannot be continued; to finish it, it is measured again from "
     "the start."),
    (RunSpec("Sweep study", ("temperature", "sweep"), name="study", stop_button="Stop sweep study…"),
     "Sweep study study is running. Shutting down stops it. What the study has measured is kept, marked unfinished; a "
     "study cannot be continued."),
    (RunSpec("Comparison", ("temperature", "one_cell"), stop_button="Stop the comparison…"),
     "Comparison is running. Shutting down stops it. A comparison writes its record only when it finishes, so nothing "
     "of this one is kept."),
    (RunSpec("Reduction map", ("reduction", None), stop_button="Stop computing…"),
     "Reduction map is running. Shutting down stops it. The map will not be shown. If its table is already written, "
     "that file stays in the reduction folder."),
    (RunSpec("Video", ("live", None), stop_button="Stop saving…", keeps_files=False),
     "Video is running. Shutting down stops it. The unfinished file is removed. The trace stays and can be saved "
     "again."),
    (RunSpec("Live simulation", ("live", None), stop_button="Stop simulation", keeps_files=False, stop_asks=False),
     "Live simulation is running. Shutting down stops it."),
    (_load(), "Prior load is running. Shutting down stops it."),
)


def _guards(request):
    """tests/conftest.py as pytest holds it: tests/ is no package, so pytest holds its conftest under a name of its
    own, and imported by name the file would be loaded a second time, beside the one whose fixtures are in force."""
    here = REPO / "tests" / "conftest.py"
    return next(plugin for plugin in request.config.pluginmanager.get_plugins()
                if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == here)


def test_a_test_that_ends_with_a_windows_request_standing_fails(request):
    """No test leaves the computer kept awake or a reason against ending the session registered, and no test that is
    not marked for the real screen reaches Windows. In every such test the session's guard stands in for the calls of
    core/gui/this_computer.py and keeps each, with what it was handed and the thread it came from; what a test's
    record leaves standing -- the computer still kept awake, a reason still registered, Windows reached -- is
    ``windows_requests_left``'s to say, and it says nothing of a record that leaves nothing. The guard's own function
    is driven here as the test runner drives it, around a test that leaves each standing and around one that leaves
    nothing: the first fail at their end, the module's record of the last request put back. The end of a test, which
    stops a run the test left live and so lets go of what the run asked, is torn down before the guard reads its
    record: it names the guard among the fixtures it needs. Under core/ no file but that module names Windows'
    calls."""
    import inspect
    from core.gui import this_computer
    guards = _guards(request)
    assert "windows_asked" in inspect.signature(guards._built_widgets_let_go.__wrapped__).parameters, (
        "the end of a test is not torn down before the guard of the requests made of Windows reads its record")
    left, window = guards.windows_requests_left, object()
    balanced = [("keep_awake", (True,), "MainThread"),
                ("block_shutdown", (window, "Training is running."), "MainThread"),
                ("block_shutdown", (window, "Training is fitting."), "MainThread"), ("flash", (window,), "MainThread"),
                ("keep_awake", (False,), "MainThread"), ("block_shutdown", (window, None), "MainThread")]
    assert (left([]), left(balanced)) == ([], [])
    assert left(balanced[:4]) == ["the computer is still kept awake",
                                  "a reason against shutting down is still registered: 'Training is fitting.'"]
    assert left(balanced + [("windows", ("kernel32.SetThreadExecutionState",), "MainThread")]) == [
        "Windows was reached by a test not marked for the real screen: kernel32.SetThreadExecutionState"]

    around_a_test = guards.windows_asked.__wrapped__       # the function the fixture was made from
    reached = "Windows was reached by a test not marked for the real screen: kernel32.SetThreadExecutionState"
    for leaves, wanted in (("nothing", None), ("awake", "the computer is still kept awake"),
                           ("reason", "a reason against shutting down is still registered: 'Training is fitting.'"),
                           ("Windows", reached)):
        run = around_a_test(request)
        calls = next(run)                                  # before the test
        assert this_computer.keep_awake(True) is True     # the test: the stand-ins, each kept
        assert this_computer.block_shutdown(window, "Training is fitting.") is True
        if leaves != "awake":
            this_computer.keep_awake(False)
        if leaves != "reason":
            this_computer.block_shutdown(window, None)
        if leaves == "Windows":
            with pytest.raises(RuntimeError):
                this_computer._windows("kernel32", "SetThreadExecutionState")
        assert all(thread == "MainThread" for _name, _args, thread in calls) and len(calls) >= 2, calls
        if wanted is None:
            with pytest.raises(StopIteration):
                next(run)                                  # after the test: nothing standing
        else:
            with pytest.raises(AssertionError, match=re.escape(wanted)):
                next(run)
        assert this_computer.state() == (False, None), "the guard did not put back the module's record"

    named = {path.relative_to(REPO).as_posix(): [call for call in _WINDOWS_CALLS
                                                 if call in path.read_text(encoding="utf-8")]
             for path in sorted((REPO / "core").rglob("*.py"))}
    assert {path: calls for path, calls in named.items() if calls} == {
        "core/gui/this_computer.py": list(_WINDOWS_CALLS)}, "a file under core/ but this_computer.py calls Windows"


def test_the_computer_is_kept_awake_from_the_start_of_a_run_to_its_end(tmp_path, windows_asked):
    """Windows is asked to keep the computer awake as a run starts, and to let it sleep again as the run ends: once
    each, both from the window's thread, however the run ends -- finished, stopped or failed. The request stands
    through the run's stop, and the run's end lets the computer sleep before its end is published."""
    from PySide6.QtCore import QObject
    from tests import _built
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held, host = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_run_host()

    def awake():
        return [(args[0], thread) for name, args, thread in windows_asked if name == "keep_awake"]

    class Ends(QObject):
        """What Windows had been asked of keeping the computer awake when each run's end was published."""

        def __init__(self):
            super().__init__()
            self.asked = []
            held.run_ended.connect(self.ended)
            _built.keep(self)

        def ended(self, _status):
            self.asked.append(awake())

    ends = Ends()
    for ending in ("finished", "stopped", "failed"):
        windows_asked.clear()
        del ends.asked[:]
        text, stand_in = RunText(held), StandIn()
        start_run(host, stand_in, spec=_training())
        assert (held.status.state, awake()) == ("starting", [(True, "MainThread")]), (ending, awake())
        stand_in.say(_LINES["budget"])
        turn_until(app, lambda: _LINES["budget"] in text.texts())
        if ending == "stopped":
            held.request_stop()
            assert held.status.state == "stopping"
        assert awake() == [(True, "MainThread")], f"the request did not stand while the run was live: {awake()}"
        if ending == "finished":
            stand_in.release()
        elif ending == "failed":
            stand_in.fail(RuntimeError("the cell file could not be read"))
        turn_until(app, lambda: not held.is_live())
        assert [status.outcome for status in text.ended] == [ending]
        assert ends.asked == [awake()] == [[(True, "MainThread"), (False, "MainThread")]], (
            f"{ending}: not asked once each way from the window's thread, or let sleep after the end was published: "
            f"{ends.asked}, {awake()}")


def test_the_reason_against_shutting_down_follows_the_runs_stage(tmp_path, windows_asked):
    """From a run's first line to its end Windows is given the reason PRISM should not be shut down, the words Windows
    lists beside PRISM: registered on the newest window built as the run is running, set again at each change of its
    stage, standing through its stop, and withdrawn as it ends, each from the window's thread; none while it starts.
    A training, and a narrowing round in its own name, says it is simulating, fitting, or saving its posterior, and
    what shutting down then loses and keeps (a fit not saved as it goes is lost whole); before its simulation, and
    every other kind of run, a live
    simulation's and a load's too, it says that shutting down stops it, then what its stop keeps, as its stop's
    question says it."""
    from tests._fixtures import build_run_host, build_window, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held = _controller_on(tmp_path, clock)
    build_window()                                     # an earlier window: the reason goes on the newest
    window, host = build_window(), build_run_host()

    def reasons():
        return [(args[1], args[0] is window, thread) for name, args, thread in windows_asked
                if name == "block_shutdown"]

    def said(text, stand_in, *lines):
        for line in lines:
            stand_in.say(line)
        turn_until(app, lambda: lines[-1] in text.texts())

    def given(*words):
        return [(word, True, "MainThread") for word in words]

    # a training through its stages, stopped in its simulation
    text, stand_in = RunText(held), StandIn(takes_stops=False)
    start_run(host, stand_in, spec=_training())
    held.tick()                                        # the controller's look, while the run starts
    assert (held.status.state, reasons()) == ("starting", []), "a reason was registered before the run's first line"
    said(text, stand_in, _LINES["budget"])
    assert reasons() == given(_RUNNING)
    said(text, stand_in, _LINES["rotation_begins"])
    assert reasons() == given(_RUNNING, _RUNNING), "the reason was not set again at the rotation"
    said(text, stand_in, _LINES["rotation_done"], _LINES["simulation_writing"])
    assert reasons() == given(_RUNNING, _RUNNING, _SIMULATING)
    held.request_stop()
    clock.advance(30)
    held.tick()
    assert (held.status.state, reasons()) == ("stopping", given(_RUNNING, _RUNNING, _SIMULATING)), (
        "the reason did not stand through the stop")
    stand_in.say("the line at which the stop is taken")
    turn_until(app, lambda: not held.is_live())
    assert reasons() == given(_RUNNING, _RUNNING, _SIMULATING, None), "the reason was not withdrawn at the end"

    # a training to its fit and the saving of its posterior; then one whose fit is not saved as it goes
    for saved, fitting, saving in ((True, _FITTING, _SAVING),
                                   (False, f"Training retrain_3 is fitting. {_LOSES_THE_FIT}",
                                    f"Training retrain_3 is saving its posterior. {_LOSES_THE_FIT}")):
        windows_asked.clear()
        text, stand_in = RunText(held), StandIn()
        start_run(host, stand_in, spec=_training())
        for line in (_LINES["budget"], _LINES["rotation_begins"], _LINES["rotation_done"],
                     _LINES["simulation_writing"], *(() if saved else (_LINES["fit_saving_off"],)),
                     _LINES["simulation_complete"], _epoch(1), _epoch(2, since=20)):   # 20 since the best: it ends
            said(text, stand_in, line)
        assert held.model.stage == "Saving"
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
        assert reasons() == given(_RUNNING, _RUNNING, _SIMULATING, fitting, saving, None), (saved, reasons())

    # a narrowing round, carried to its fit: a training's words, in the round's own name
    windows_asked.clear()
    text, stand_in = RunText(held), StandIn(word="Narrowing round")
    start_run(host, stand_in, spec=_training(word="Narrowing round", name="round_2",
                                             facts={**_training().facts, "narrowed": True}))
    for line in (_LINES["budget"], _LINES["rotation_begins"], _LINES["rotation_done"], _LINES["simulation_writing"],
                 _LINES["simulation_complete"], _epoch(1)):
        said(text, stand_in, line)
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    assert reasons() == given(*(words.replace("Training retrain_3", "Narrowing round round_2")
                                for words in (_RUNNING, _RUNNING, _SIMULATING, _FITTING)), None), reasons()

    # every other kind of run
    for spec, reason in _OTHER_REASONS:
        windows_asked.clear()
        text, stand_in = RunText(held), StandIn(word=spec.word)
        start_run(host, stand_in, spec=spec)
        assert reasons() == [], spec.word
        said(text, stand_in, "a line of the run")
        assert reasons() == given(reason), spec.word
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
        assert reasons() == given(reason, None), spec.word


# ── a run's end, in words ──
def _ended_record(**changed) -> dict:
    """A training's history as the window writes it at the training's end: stopped on Sun 11 Oct at 12:31, in its
    fit, its 20,480,000 simulations kept and its training kept up to epoch 62; ``changed`` sets any field, and a
    field of ``fit`` or ``simulation_set`` it does not name reads None."""
    record = {"id": "20261008T190211-training", "word": "Training", "name": "retrain_3",
              "place": ["inference", "training"], "records_root": "records", "started": _at(8, 19, 2, 11),
              "ended": _at(11, 12, 31), "last_heard": None, "outcome": "stopped", "stop_reason": None,
              "stage": "Fitting", "acknowledged": False,
              "tries": [{"started": _at(8, 19, 2, 11), "ended": _at(11, 12, 31), "outcome": "stopped",
                         "failure": None, "batches_before": 0, "batches_after": 10_000, "epochs_before": None,
                         "epochs_after": 62}],
              "simulation_set": {"folder": _FOLDER, "batches": 10_000, "at_start": 0, "at_end": 10_000,
                                 "per_batch": 2_048, "seconds_per_batch": 9.7},
              "fit": {"epochs": 62, "best_epoch": 60, "ended_by": None, "simulations": 20_480_000, "kept_id": "k1",
                      "passes": 62, "expected": None, "discarded": False},
              "result": None, "failure": None}
    for field in ("fit", "simulation_set"):
        if isinstance(changed.get(field), dict):
            changed = {**changed, field: {**{name: None for name in record[field]}, **changed[field]}}
    record.update(changed)
    return record


def _ended_try(ended, outcome, batches) -> list:
    """A run's one try, begun on Thu 8 Oct at 19:02, as its history keeps it at the run's end."""
    return [{"started": _at(8, 19, 2, 11), "ended": ended, "outcome": outcome, "failure": None, "batches_before": 0,
             "batches_after": batches, "epochs_before": None, "epochs_after": None}]


_FINISHED = dict(outcome="finished", stage="Saving", started=_at(8, 19, 10), ended=_at(14, 14, 10),
                 tries=_ended_try(_at(14, 14, 10), "finished", 10_000),
                 result={"kind": "posterior", "id": "q1", "name": "retrain_3", "asked": "retrain_3"})
_SAVED = "Training finished on Wed 14 Oct, 14:10, after 5 days 19 hours. The posterior retrain_3 is saved."
_FIT_KEPT = "The 20,480,000 simulations are kept; the training is kept up to epoch 62, so you can resume it."
_MEASURED = ("The record cell_a_3oct is kept, marked unfinished, with what it has measured so far. A measurement "
             "cannot be continued; to finish it, it is measured again from the start.")
# How each kind of end is said: what its history holds beyond ``_ended_record``'s, the moment it is said at, whether
# a stop came too late, then its sentence and what the foot says was kept.
_ENDS = {
    "finished": (_FINISHED, _at(14, 14, 10), False, _SAVED, ""),
    "finished a week and more ago": (
        _FINISHED, _at(22, 9, 0), False,
        "Training finished on 14 Oct 2026, 14:10, after 5 days 19 hours. The posterior retrain_3 is saved.", ""),
    "finished, saved without its name": (
        {**_FINISHED, "result": {"kind": "posterior", "id": "q1", "name": "", "asked": "retrain_3"}}, _at(14, 14, 10),
        False, "Training finished on Wed 14 Oct, 14:10, after 5 days 19 hours. The posterior is saved. It was saved "
               "without its name: 'retrain_3' was taken while the run was going.", ""),
    "finished, the fit ended earlier than its estimate": (
        {**_FINISHED, "fit": {"epochs": 84, "best_epoch": 64, "ended_by": "patience", "expected": 130}},
        _at(14, 14, 10), False, f"{_SAVED} The fit ended at epoch 84, earlier than the estimate of about 130: "
                                f"validation had not improved for 20 epochs (best at epoch 64).", ""),
    "finished, the fit ended later than its estimate": (
        {**_FINISHED, "fit": {"epochs": 140, "best_epoch": 120, "ended_by": "patience", "expected": 130}},
        _at(14, 14, 10), False, f"{_SAVED} The fit ended at epoch 140: validation had not improved for 20 epochs "
                                f"(best at epoch 120).", ""),
    "finished, the fit ended with no estimate": (
        {**_FINISHED, "fit": {"epochs": 84, "best_epoch": 64, "ended_by": "patience"}}, _at(14, 14, 10), False,
        f"{_SAVED} The fit ended at epoch 84: validation had not improved for 20 epochs (best at epoch 64).", ""),
    "finished, the fit stopped at its limit": (
        {**_FINISHED, "fit": {"epochs": 51, "best_epoch": 49, "ended_by": "limit", "expected": 130}},
        _at(14, 14, 10), False, f"{_SAVED} The fit stopped at its epoch limit after 51 epochs, before it had stopped "
                                f"improving. The saved network is the last epoch's, not the best (epoch 49).", ""),
    "finished, a stop asked too late": (
        _FINISHED, _at(14, 14, 10), True, f"{_SAVED} A stop was asked for, but the run had already finished.", ""),
    "stopped during the fit": (
        {}, _at(11, 12, 31), False, f"Training was stopped on Sun 11 Oct, 12:31, during the fit. {_FIT_KEPT}",
        _FIT_KEPT),
    "stopped during the fit, its training discarded": (
        {"fit": {"epochs": 62, "best_epoch": 60, "simulations": 20_480_000, "kept_id": "k1", "passes": 62,
                 "discarded": True}}, _at(11, 12, 31), False,
        "Training was stopped on Sun 11 Oct, 12:31, during the fit. The 20,480,000 simulations are kept; the training "
        "was discarded, as you asked.",
        "The 20,480,000 simulations are kept; the training was discarded, as you asked."),
    "stopped during the simulation": (
        {"stage": "Simulation", "fit": None, "tries": _ended_try(_at(11, 12, 31), "stopped", 6_150)}, _at(11, 12, 31),
        False, "Training was stopped on Sun 11 Oct, 12:31, during simulation. The 6,150 batches simulated so far are "
               "kept; you can continue from batch 6,150.",
        "The 6,150 batches simulated so far are kept; you can continue from batch 6,150."),
    "stopped during the rotation": (
        {"stage": "Rotation", "fit": None, "simulation_set": None, "tries": _ended_try(_at(11, 12, 31), "stopped", 0)},
        _at(11, 12, 31), False, "Training was stopped on Sun 11 Oct, 12:31, during the rotation. Nothing was "
                                "simulated.", "Nothing was simulated."),
    "stopped when Windows shut down": (
        {"stop_reason": "Windows shut down"}, _at(11, 12, 31), False,
        f"Training was stopped on Sun 11 Oct, 12:31, when Windows shut down. {_FIT_KEPT}", _FIT_KEPT),
    "stopped when Windows shut down, PRISM ended before the run": (
        {"stop_reason": "Windows shut down", "last_heard": _at(11, 12, 33), "fit": {"simulations": 20_480_000},
         "tries": _ended_try(_at(11, 12, 31), "stopped", 9_950)}, _at(11, 12, 40), False,
        "Training was stopped on Sun 11 Oct, 12:31, when Windows shut down. What it had saved by then is kept.",
        "What it had saved by then is kept."),
    "stopped, in a history written before the stop's reason was kept": (
        {"stop_reason": "left out"}, _at(11, 12, 31), False,
        f"Training was stopped on Sun 11 Oct, 12:31, during the fit. {_FIT_KEPT}", _FIT_KEPT),
    "failed": (
        {"outcome": "failed", "stage": "Simulation", "fit": None, "tries": _ended_try(_at(11, 12, 31), "failed", 6_150),
         "failure": {"type": "OutOfMemoryError", "message": "CUDA out of memory.", "kind": "out of memory",
                     "reopen": False}}, _at(11, 12, 31), False,
        "Training failed on Sun 11 Oct, 12:31, during simulation: the graphics card ran out of memory. The "
        "simulations up to batch 6,150 are kept.", "The simulations up to batch 6,150 are kept."),
    "failed, its tries ended when Windows shut down": (
        {"outcome": "failed", "stop_reason": "Windows shut down", "stage": "Simulation", "fit": None,
         "tries": _ended_try(_at(11, 12, 31), "failed", 6_150),
         "failure": {"type": "OutOfMemoryError", "message": "CUDA out of memory.", "kind": "out of memory",
                     "reopen": False}}, _at(11, 12, 31), False,
        "Training failed on Sun 11 Oct, 12:31, during simulation: the graphics card ran out of memory. The "
        "simulations up to batch 6,150 are kept.", "The simulations up to batch 6,150 are kept."),
    "not seen to end": (
        {"outcome": "not seen to end", "stage": "Simulation", "ended": None, "last_heard": _at(3, 3, 12), "fit": None,
         "simulation_set": {"folder": _FOLDER, "batches": 10_000, "at_start": 0, "at_end": 6_150},
         "tries": _ended_try(None, None, 6_150)}, _at(8, 19, 2), False,
        "PRISM closed while this training was running; last heard Sat 3 Oct, 03:12, simulation, batch 6,150 of "
        "10,000.", "What it had saved by then is kept."),
    "a measurement finished": (
        {"word": "Measurement", "name": "cell_a_3oct", "place": ["temperature", "one_cell"], "outcome": "finished",
         "stage": "Driven", "started": _at(11, 11, 19), "fit": None, "simulation_set": None,
         "result": {"kind": "fdt", "id": "f1", "name": "cell_a_3oct", "asked": "cell_a_3oct"}}, _at(11, 12, 31),
        False, "Measurement finished on Sun 11 Oct, 12:31, after 1 hour 12 minutes. The record cell_a_3oct is saved.",
        ""),
    "a measurement stopped": (
        {"word": "Measurement", "name": "cell_a_3oct", "place": ["temperature", "one_cell"], "stage": "Driven",
         "fit": None, "simulation_set": None}, _at(11, 12, 31), False,
        f"Measurement was stopped on Sun 11 Oct, 12:31. {_MEASURED}", _MEASURED),
    "a check finished, under the name asked for": (
        {"word": "Check", "name": "Calibration", "place": ["inference", "checks"], "outcome": "finished",
         "stage": None, "started": _at(11, 12, 19), "fit": None, "simulation_set": None,
         "result": {"kind": "calibration", "id": "c1", "name": "cal_oct", "asked": "cal_oct"}}, _at(11, 12, 31),
        False, "Check finished on Sun 11 Oct, 12:31, after 12 minutes. The calibration cal_oct is saved.", ""),
    "a check finished, asked for no name": (
        {"word": "Check", "name": "Calibration", "place": ["inference", "checks"], "outcome": "finished",
         "stage": None, "started": _at(11, 12, 19), "fit": None, "simulation_set": None,
         "result": {"kind": "calibration", "id": "c1", "name": "", "asked": ""}}, _at(11, 12, 31), False,
        "Check finished on Sun 11 Oct, 12:31, after 12 minutes. The calibration is saved.", ""),
    "a sweep study finished, which names no record": (
        {"word": "Sweep study", "name": "sweep_a", "place": ["temperature", "sweep"], "outcome": "finished",
         "stage": None, "started": _at(11, 9, 1), "fit": None, "simulation_set": None, "result": None},
        _at(11, 12, 31), False, "Sweep study finished on Sun 11 Oct, 12:31, after 3 hours 30 minutes.", ""),
    "stopped during the fit, no training kept": (
        {"fit": {"epochs": 3, "best_epoch": 2, "simulations": 20_480_000}}, _at(11, 12, 31), False,
        "Training was stopped on Sun 11 Oct, 12:31, during the fit. The 20,480,000 simulations are kept.",
        "The 20,480,000 simulations are kept."),
    "stopped as its fit ended, before its posterior was written": (
        {"stage": "Saving"}, _at(11, 12, 31), False,
        "Training was stopped on Sun 11 Oct, 12:31, as its fit ended. The training is kept with its fit complete; "
        "resuming it writes the posterior.", "The training is kept with its fit complete; resuming it writes the "
                                             "posterior."),
    "failed during the fit, its training kept": (
        {"outcome": "failed", "tries": _ended_try(_at(11, 12, 31), "failed", 10_000),
         "failure": {"type": "RuntimeError", "message": "the fit's epoch 63 failed", "kind": "out of memory",
                     "reopen": True}}, _at(11, 12, 31), False,
        "Training failed on Sun 11 Oct, 12:31, during the fit: the graphics card ran out of memory. The simulations "
        "and the training up to epoch 62 are kept.", "The simulations and the training up to epoch 62 are kept."),
    "failed, its message ending in a full stop": (
        {"outcome": "failed", "stage": "Simulation", "fit": None, "tries": _ended_try(_at(11, 12, 31), "failed", 6_150),
         "failure": {"type": "RuntimeError", "message": "The simulation of batch 6151 failed.\nIts second line.",
                     "kind": "other", "reopen": False}}, _at(11, 12, 31), False,
        "Training failed on Sun 11 Oct, 12:31, during simulation: The simulation of batch 6151 failed. The "
        "simulations up to batch 6,150 are kept.", "The simulations up to batch 6,150 are kept."),
    "failed, the error with no message": (
        {"outcome": "failed", "stage": "Simulation", "fit": None, "tries": _ended_try(_at(11, 12, 31), "failed", 6_150),
         "failure": {"type": "RuntimeError", "message": "", "kind": "other", "reopen": False}}, _at(11, 12, 31),
        False, "Training failed on Sun 11 Oct, 12:31, during simulation: the error gave no message. The simulations "
               "up to batch 6,150 are kept.", "The simulations up to batch 6,150 are kept."),
    "not seen to end in its fit": (
        {"outcome": "not seen to end", "ended": None, "last_heard": _at(3, 3, 12),
         "tries": _ended_try(None, None, 10_000)}, _at(8, 19, 2), False,
        "PRISM closed while this training was running; last heard Sat 3 Oct, 03:12, the fit, epoch 62.",
        "What it had saved by then is kept."),
    "not seen to end in its rotation": (
        {"outcome": "not seen to end", "stage": "Rotation", "ended": None, "last_heard": _at(3, 3, 12), "fit": None,
         "simulation_set": None, "tries": _ended_try(None, None, 0)}, _at(8, 19, 2), False,
        "PRISM closed while this training was running; last heard Sat 3 Oct, 03:12, the rotation.",
        "What it had saved by then is kept."),
    "not seen to end as its posterior was saved": (
        {"outcome": "not seen to end", "stage": "Saving", "ended": None, "last_heard": _at(3, 3, 12),
         "tries": _ended_try(None, None, 10_000)}, _at(8, 19, 2), False,
        "PRISM closed while this training was running; last heard Sat 3 Oct, 03:12, saving its posterior.",
        "What it had saved by then is kept."),
    "a measurement not seen to end": (
        {"word": "Measurement", "name": "cell_a_3oct", "place": ["temperature", "one_cell"],
         "outcome": "not seen to end", "stage": "Driven", "ended": None, "last_heard": _at(3, 3, 12), "fit": None,
         "simulation_set": None}, _at(8, 19, 2), False,
        "PRISM closed while this measurement was running; last heard Sat 3 Oct, 03:12, driven.", ""),
    "not seen to end, at a moment not known": (
        {"outcome": "not seen to end", "stage": "Simulation", "ended": None, "last_heard": None, "fit": None,
         "simulation_set": {"folder": _FOLDER, "batches": 10_000, "at_start": 0, "at_end": 6_150},
         "tries": _ended_try(None, None, 6_150)}, _at(8, 19, 2), False,
        "PRISM closed while this training was running; last heard at a moment not known, simulation, batch 6,150 of "
        "10,000.", "What it had saved by then is kept."),
    "a prior build stopped": (
        {"word": "Prior build", "name": "", "place": ["inference", "prior"], "stage": None, "fit": None,
         "simulation_set": None}, _at(11, 12, 31), False,
        "Prior build was stopped on Sun 11 Oct, 12:31. Nothing of the prior is kept; it is built again from the start "
        "next time.", "Nothing of the prior is kept; it is built again from the start next time."),
}


@pytest.mark.parametrize("case", list(_ENDS))
def test_each_end_says_how_and_when_and_never_finished_after_a_stop_or_a_failure(case):
    """How a run ended is said from its history alone, in the words its page, the foot and Home say: when it
    finished and after how long, the record it saved and that it was saved without its name, how its fit ended, a
    stop that came too late; when it was stopped and where it was, or that Windows shut down, and what the stop kept,
    a training's by its stage and any other run's as its stop's question said it; when it failed, where and why, and
    what was kept; and a run PRISM did not see end, by when it was last heard and where it was. A date within seven
    days carries its weekday. A history written before the stop's reason was kept is read as one with none. Nothing
    says "finished" after a stop or a failure, and no line of the window says a measurement finished whatever its
    end."""
    from core.gui.run import ends
    from core.gui.worker import CANCEL_TOO_LATE_TEXT
    changed, now, too_late, sentence, kept = _ENDS[case]
    record = _ended_record(**changed)
    if record["stop_reason"] == "left out":
        del record["stop_reason"]
    assert ends.words(record, now, too_late=too_late) == (sentence, kept)
    assert CANCEL_TOO_LATE_TEXT is ends.TOO_LATE_TEXT, "the run's text and its page say a late stop in two words"
    if record["outcome"] != "finished":
        assert not re.search(r"\bfinished\b", sentence), sentence
    said = [path.relative_to(REPO).as_posix() for path in (REPO / "core").rglob("*.py")
            if "FDT run finished" in path.read_text(encoding="utf-8")]
    assert said == [], f"a line of the window says a measurement finished, whatever its end: {said}"


def test_the_end_is_said_from_what_the_controller_heard_of_the_run(tmp_path, monkeypatch):
    """The end's words are said from the history the controller writes of what it heard. A training whose call was
    handed the name to save its posterior under returns the posterior it saved, after a fit that ended by its
    patience earlier than its estimate, a stop asked of it too late to stop anything: it is said finished, after how
    long, with its posterior, how its fit ended and the late stop; and its history holds the record it wrote with
    the name asked, how its fit ended, the epochs it was expected to run and its stages. A check whose call was
    handed a name returns its calibration under that name and is said with it; one handed none, as the window
    starts a check, is said saved, and never saved without its name: a run's own word for itself is no name. A
    measurement's name is the name of the record its page made at the click, which its call is handed."""
    import logging
    import threading
    from core.gui.run import controller as controller_module
    from core.gui.run import history
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, start_run, turn_until
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held = _controller_on(tmp_path, clock)
    monkeypatch.setattr(controller_module, "_expected", lambda store, facts: _EXPECTED)
    asked, log = threading.Event(), logging.getLogger("core.SBI.train")

    def training(*, name, **_kwargs):
        for line in (_LINES["budget"], _LINES["rotation_begins"], _LINES["rotation_done"],
                     _LINES["simulation_writing"], _LINES["simulation_complete"], _epoch(1), _epoch(2), _epoch(3, 1)):
            log.info(line)
        asked.wait(10)
        try:
            raise RuntimeError("a last pass the fit recovered from")
        except RuntimeError:
            log.info(_epoch(4, 2))          # said inside a handler, the stop is carried: the fit ends by its patience
        return SimpleNamespace(kind="posterior", id="q1", name=name)

    start_run(build_run_host(), training, spec=_training(patience=2), name="retrain_3")
    turn_until(app, lambda: len(held.model.epochs()) == 3)
    held.request_stop()
    clock.advance(4 * 3600 + 12 * 60)
    asked.set()
    turn_until(app, lambda: not held.is_live())
    record = history.read(tmp_path / "runs" / f"{held.status.run}.json")
    assert held.status.sentence == (
        "Training finished on Thu 8 Oct, 23:14, after 4 hours 12 minutes. The posterior retrain_3 is saved. The fit "
        "ended at epoch 4, earlier than the estimate of about 130: validation had not improved for 2 epochs (best at "
        "epoch 2). A stop was asked for, but the run had already finished."), held.status.sentence
    assert (record["result"], record["fit"]["ended_by"], record["fit"]["expected"], record["stages"]) == (
        {"kind": "posterior", "id": "q1", "name": "retrain_3", "asked": "retrain_3"}, "patience", 130,
        [[stage.name, list(stage.begins), stage.counts] for stage in _TRAINING]), record

    check = RunSpec("Check", ("inference", "checks"), name="Calibration", stop_button="Stop the calibration…",
                    kinds=("calibration",))                 # as the check's page names it: its word, no record's name
    measurement = RunSpec("Measurement", ("temperature", "one_cell"), name="cell_a_3oct",
                          stop_button="Stop measurement…", kinds=("fdt",))
    for spec, given, kind, returned, saved in (
            (check, {"name": "cal_oct"}, "calibration", "cal_oct", "The calibration cal_oct is saved."),
            (check, {}, "calibration", "", "The calibration is saved."),
            (measurement, {"writer": SimpleNamespace(name="cell_a_3oct", id="f1")}, "fdt", "cell_a_3oct",
             "The record cell_a_3oct is saved.")):
        start_run(build_run_host(), lambda **_kwargs: SimpleNamespace(kind=kind, id="c1", name=returned), spec=spec,
                  **given)
        turn_until(app, lambda: not held.is_live())
        record = history.read(tmp_path / "runs" / f"{held.status.run}.json")
        assert (held.status.sentence, record["result"]) == (
            f"{spec.word} finished on Thu 8 Oct, 23:14, after less than a minute. {saved}",
            {"kind": kind, "id": "c1", "name": returned, "asked": returned}), held.status.sentence


def test_an_end_waiting_to_be_seen_is_set_aside_only_by_a_run_that_leaves_an_end_of_its_own(tmp_path, monkeypatch):
    """A run started while an end waits to be seen sets that end aside, written seen in its history, only once it has
    ended with an end of its own to say: a run that keeps a history file (a training, a narrowing round, a prior's
    build, a check, an inference, a measurement, a sweep study, a comparison, a reduction map). A load and a video,
    which keep no file and end into idle, set nothing aside, nor does a start taken back before its worker was
    queued, nor a run the stage refused before anything was spent: after each, the end that waited is the status
    again, its history still unseen."""
    from core.gui.run import history
    from core.refusals import Refusal
    from tests._fixtures import asked as box_asked, build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    held, host = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11))), build_run_host()
    first = StandIn()
    start_run(host, first, spec=_training())
    first.release()
    turn_until(app, lambda: not held.is_live())
    waiting = held.status
    path = tmp_path / "runs" / f"{waiting.run}.json"

    def kept_waiting(what):
        assert (held.status == waiting, history.read(path)["acknowledged"]) == (True, False), (
            f"{what} set aside the end that waited to be seen", held.status)

    for spec in (_load(), RunSpec("Video", ("live", None), stop_button="Stop saving…", keeps_files=False)):
        stand_in = StandIn(word=spec.word)
        start_run(host, stand_in, spec=spec)
        stand_in.release()
        turn_until(app, lambda: not held.is_live())
        kept_waiting(spec.word)
    with pytest.raises(TypeError):
        start_run(host, StandIn(), spec=_training(), watch_dir=5)
    kept_waiting("a start taken back")
    refused = StandIn()
    start_run(host, refused, spec=_training())
    refused.fail(Refusal("The posterior name 'retrain_3' is taken.", field="name"))
    turn_until(app, lambda: not held.is_live())
    assert box_asked().sentence == "The posterior name 'retrain_3' is taken."
    kept_waiting("a run refused before anything was spent")
    later = StandIn()
    start_run(host, later, spec=_training())
    later.release()
    turn_until(app, lambda: not held.is_live())
    assert (held.status.run != waiting.run, held.status.unacknowledged_end, history.read(path)["acknowledged"]) == (
        True, True, True), "a run that left an end of its own did not set aside the end that waited"


def test_at_the_end_the_run_lets_go_of_everything_and_is_published_last(tmp_path, windows_asked, monkeypatch):
    """At a run's end, in this order: its history says how it ended, every panel's controls are free again, the
    computer may sleep again and the reason against ending the session is withdrawn, as the guard of the requests
    made of Windows records them; only then is the run's status published, saying how and when the run ended, then
    its end; then the taskbar flashes on the newest window built. Every step is taken on the window's thread."""
    import threading
    from PySide6.QtCore import QObject
    from core.gui.panels.base_panel import BasePanel
    from core.gui.run import history
    from tests import _built
    from tests._fixtures import build_window, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    window = build_window()
    host = window.part("inference").posterior_panel
    order = windows_asked                             # the guard's record, with the end's other steps put in it
    write, free = history.write, BasePanel.set_controls_enabled

    def written(folder, record):
        order.append(("history", (record["outcome"],), threading.current_thread().name))
        return write(folder, record)

    def controls(panel, enabled):
        order.append(("controls", (enabled,), threading.current_thread().name))
        return free(panel, enabled)

    monkeypatch.setattr(history, "write", written)
    monkeypatch.setattr(BasePanel, "set_controls_enabled", controls)

    class Heard(QObject):
        """Each status and each end the run's controller publishes, put in the record as it is heard."""

        def __init__(self):
            super().__init__()
            held.status_changed.connect(self.status)
            held.run_ended.connect(self.ended)
            _built.keep(self)

        def status(self, status):
            order.append(("status", (status.state,), threading.current_thread().name))

        def ended(self, status):
            order.append(("run_ended", (status.outcome,), threading.current_thread().name))

    heard = Heard()
    text, stand_in = RunText(held), StandIn()
    start_run(host, stand_in, spec=_training())
    stand_in.say(_LINES["budget"])
    turn_until(app, lambda: _LINES["budget"] in text.texts())
    del order[:]
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    steps = []
    for name, args, thread in order:
        assert thread == "MainThread", (name, thread)
        step = (name, args[0] is window if name == "flash" else args[-1])
        if not steps or steps[-1] != step:
            steps.append(step)
    assert steps == [("history", "finished"), ("controls", True), ("keep_awake", False), ("block_shutdown", None),
                     ("status", "ended"), ("run_ended", "finished"), ("flash", True)], steps
    assert held.status.sentence == "Training finished on Thu 8 Oct, 19:02, after less than a minute.", (
        held.status.sentence)


def test_the_taskbar_flashes_for_an_end_and_for_needs_you(tmp_path, windows_asked):
    """The taskbar flashes on the newest window built, from the window's thread, once a run has come to need someone,
    after the status that says so is published, and not again while it still does, a stop asked meanwhile
    included; and once at an end that is to be seen, after its status. Never at the end of a run that ends straight into idle (one that keeps no file, or only
    loads a record), nor at a refusal made before anything was spent."""
    import threading
    from PySide6.QtCore import QObject
    from core.gui.run.model import SILENCE_NEEDS_YOU_S
    from core.refusals import Refusal
    from tests import _built
    from tests._fixtures import build_run_host, build_window, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held = _controller_on(tmp_path, clock)
    window, host = build_window(), build_run_host()

    class Heard(QObject):
        """Whether each status the run's controller publishes says the run needs someone, put in the guard's record
        as it is heard."""

        def __init__(self):
            super().__init__()
            held.status_changed.connect(self.status)
            _built.keep(self)

        def status(self, status):
            windows_asked.append(("status", (bool(status.needs_you),), threading.current_thread().name))

    def said():
        return [(name, args[0] is window if name == "flash" else args[0], thread)
                for name, args, thread in windows_asked if name in ("flash", "status")]

    def flashes():
        return [step for step in said() if step[0] == "flash"]

    heard = Heard()
    text, stand_in = RunText(held), StandIn()
    start_run(host, stand_in, spec=_training())
    for line in (_LINES["budget"], _LINES["rotation_begins"], _LINES["rotation_done"], _LINES["simulation_writing"]):
        stand_in.say(line)
    turn_until(app, lambda: _LINES["simulation_writing"] in text.texts())
    windows_asked.clear()
    clock.advance(SILENCE_NEEDS_YOU_S + 1)
    held.tick()
    held.tick()
    assert held.status.needs_you, "the run does not need someone after its silence"
    assert flashes() == [("flash", True, "MainThread")], said()
    assert said().index(("status", True, "MainThread")) < said().index(("flash", True, "MainThread")), said()
    windows_asked.clear()
    held.request_stop()                               # its status changes while it still needs someone
    assert (held.status.state, bool(held.status.needs_you), flashes()) == ("stopping", True, []), said()
    turn_until(app, lambda: not held.is_live())
    assert said()[-2:] == [("status", False, "MainThread"), ("flash", True, "MainThread")], said()
    assert len(flashes()) == 1, said()
    held.acknowledge(held.status.run)                 # seen: each run below starts with no end waiting
    for spec, refused in ((RunSpec("Live simulation", ("live", None), keeps_files=False), False), (_load(), False),
                          (_training(), True)):
        windows_asked.clear()
        stand_in = StandIn(word=spec.word)
        start_run(host, stand_in, spec=spec)
        if refused:
            stand_in.fail(Refusal("The posterior name 'retrain_3' is taken.", field="name"))
        else:
            stand_in.release()
        turn_until(app, lambda: not held.is_live())
        assert (held.status.state, flashes()) == ("idle", []), (spec.word, refused, said())


# ── the run's page: in the place of a page's results, drawn from the run's model ──
def _page_says(view) -> dict:
    """What the run's page shows, read off its parts, each only while it is in sight: its sentence, its stages as
    markers and words, the four moments of its time axis and the two sides of the line under it, the line that names
    what follows, the rows of its health table as written in their cells, and its messages, each a marker and its
    words."""
    def shown(part):
        return part.isVisibleTo(view)

    table = view.health.view.model()
    return {"sentence": view.sentence.text(), "stages": view.stage_line.words() if shown(view.stage_line) else (),
            "axis": view.axis._times if shown(view.axis) else None,
            "caption": ((view.axis_caption.started.text(), view.axis_caption.words.text())
                        if shown(view.axis_caption) else None),
            "follows": view.follows.text() if shown(view.follows) else "",
            "health": [tuple(table.index(row, column).data().strip() for column in range(3))
                       for row in range(table.rowCount())] if shown(view.health) else [],
            "messages": ([(row.kind, row.sentence.plain_text()) for row in view.message_rows()]
                         if shown(view.messages) else [])}


def _model_says(model, now, right) -> dict:
    """The same, as the run's model reads it at ``now``: the time axis's right-hand words are ``right``."""
    from core.gui.run.model import LEADS
    from core.gui.widgets.message_row import opening
    axis = model.axis()
    return {"sentence": model.sentence(), "stages": tuple((word.name, word.marker, word.word)
                                                          for word in model.stage_line()),
            "axis": None if axis is None else (axis.start, axis.now, axis.end_from, axis.end_by),
            "caption": None if axis is None else (f"Started {texts.date_text(axis.start, axis.now)}", right),
            "follows": "" if axis is None else axis.follows,
            "health": [(row.measure, row.now, row.state) for row in model.health()],
            "messages": [(message.kind, f"{opening(LEADS[message.kind], message.when, now)} {message.sentence}")
                         for message in model.messages()]}


def test_a_run_takes_the_place_of_the_pages_results_and_gives_it_back(tmp_path, monkeypatch):
    """While a page's run shows, the run's page stands in the right-hand column in the place of the page's
    figures, which it holds in sight under its messages; the form stays at the left, in sight and locked. A run
    that finishes gives the column back, the figures in their place at the heights they had and "How the run went"
    above them, which opens the run's page again, ended. A run that is stopped or fails keeps its page, ended, until
    "Back to the form", and the layout saved meanwhile keeps the heights the figures had in their place; a failure
    is said on that page, never in a box. A run that only loads a record never takes the column nor the page that
    shows there: after a finish "How the run went" stays above the figures and opens the finished run's page, and a
    stopped run's page stays, its sentence and its full text with it, while a load runs and after. A refusal made
    before anything was spent gives the column back with no link. A run that is over is drawn again from its two
    files alone, past."""
    from core.refusals import Refusal
    from core.gui import settings
    from core.gui.run import history
    from tests._fixtures import all_asked, build_run_host, choose, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host = build_run_host()
    view, text, split = host.run_view, RunText(held), host.results_split
    host.figure_stack.add_figure("a figure", b"")
    shown_off_screen(host, 1000, 800)
    store, key = settings.settings(), f"layout/{host.layout_key()}/results"

    def column():
        """Which page the column shows, whether the run's page holds the figures in sight under its messages, the
        link above the figures, and the form's state: in sight, and free."""
        return (host.results.currentWidget().objectName(),
                view.isAncestorOf(host.figure_stack) and view.figures.isVisibleTo(view),
                view.how_link.isVisibleTo(host), host.controls_scroll.isVisibleTo(host), host.controls.isEnabled())

    def page():
        return (view.mode, view.stop_button.isVisibleTo(view), view.back_button.isVisibleTo(view), view.sentence.text())

    def ended(stand_in, how):
        how(stand_in)
        turn_until(app, lambda: not held.is_live())

    assert column() == ("figures", False, False, True, True)
    finished = StandIn()
    start_run(host, finished, spec=_training(stop_button="Stop training…"))
    assert column() == ("run", True, False, True, False), "the run's page did not take the column at the start"
    assert page() == ("live", True, False, "Training retrain_3 is starting.")
    assert view.stop_button.text() == "Stop training…" and host.results_split.indexOf(host.figure_stack) == -1
    ended(finished, StandIn.release)
    assert column() == ("figures", False, True, True, True), "a finished run kept the column"
    assert split.indexOf(host.figure_stack) == 0, "the figures did not go back to their place"
    loads = StandIn(word="Prior load")
    start_run(host, loads, spec=_load())
    assert column() == ("figures", False, True, True, False), "a load took the column, or hid 'How the run went'"
    ended(loads, StandIn.release)
    assert column() == ("figures", False, True, True, True), "a load's end hid 'How the run went'"
    view.how_link.click()
    assert column() == ("run", True, False, True, True)
    assert page() == ("ended", False, True, "Training finished on Thu 8 Oct, 19:02, after less than a minute.")
    view.back_button.click()
    assert column() == ("figures", False, True, True, True)

    split.setSizes([sum(split.sizes()) - 150, 150])         # the figures and the page's own lines, as a person set them
    heights = split.sizes()
    host.save_layout(store)
    saved = store.value(key)
    choose(monkeypatch, words="Stop training")                # the stop's question, answered
    for how, said, load in ((lambda stand_in: view.stop_button.click(),
                             "Training was stopped on Thu 8 Oct, 19:02, as it started. Nothing was simulated.", True),
                            (lambda stand_in: stand_in.fail(RuntimeError("the run broke")),
                             "Training retrain_3 failed.", False)):
        stand_in = StandIn()
        start_run(host, stand_in, spec=_training())
        stand_in.say(_LINES["budget"])
        turn_until(app, lambda: held.status.state == "running")
        ended(stand_in, how)
        assert column() == ("run", True, False, True, True), f"the page went before 'Back to the form': {said}"
        assert page() == ("ended", False, True, said)
        host.save_layout(store)
        assert store.value(key) == saved, f"the layout saved while the run's page held the figures lost them: {said}"
        if load:                                        # a load started from the same page, the run's page in sight
            kept, loads = view.full_text.lines(), StandIn(word="Prior load")
            start_run(host, loads, spec=_load())
            during = (column(), page(), view.full_text.lines() == kept)
            ended(loads, StandIn.release)
            assert during == (("run", True, False, True, False), ("ended", False, True, said), True), (
                "a load took the stopped run's page")
            assert (column(), page(), view.full_text.lines() == kept) == (
                ("run", True, False, True, True), ("ended", False, True, said), True), (
                "a load's end lost the stopped run's page")
        view.back_button.click()
        assert column() == ("figures", False, True, True, True), said
        assert (split.indexOf(host.figure_stack), split.sizes()) == (0, heights), (
            f"the figures did not go back to their place at the heights they had: {said}")
    assert [box.title for box in all_asked()] == ["Stop training?"], "the failure was told in a box"
    assert (view.failure.isVisibleTo(view), view.failure_message.text()) == (True, "the run broke"), (
        "the failure is not said on its page")
    last = held.status.run
    record = history.read(tmp_path / "runs" / f"{last}.json")
    live_text = view.full_text.lines()
    view.show_past(record, tmp_path / "runs" / f"{last}.txt", clock=held.now)
    assert page() == ("past", False, True, "Training failed on Thu 8 Oct, 19:02, before its simulation: the run broke. "
                                           "No simulation was saved.")
    assert view.full_text.lines() == live_text, "the past page's full text is not its file's"

    refused = StandIn()
    start_run(host, refused, spec=_training())
    assert column()[0] == "run"
    ended(refused, lambda stand_in: stand_in.fail(Refusal("The posterior name 'retrain_3' is taken.", field="name")))
    assert column() == ("figures", False, False, True, True), "a refusal before anything was spent kept the column"
    assert [status.outcome for status in text.ended] == ["finished", "finished", "stopped", "finished", "failed", None]


def test_how_the_run_went_stands_above_the_figures_at_the_looks_gap(tmp_path):
    """"How the run went" stands above the figures it opens the run's page from at the gap the look sets between a
    heading and the block under it, at the ordinary text size and at a larger one."""
    from core.gui import design
    from tests._fixtures import build_run_host, qt_app
    from tests._look_helpers import text_size
    from tests._runs import Clock, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    gaps = []
    for factor in (1.0, 1.5):
        with text_size(factor):
            host, stand_in = build_run_host(), StandIn()
            shown_off_screen(host, 1000, 800)
            start_run(host, stand_in, spec=_training())
            stand_in.release()
            turn_until(app, lambda: not held.is_live())
            app.processEvents()
            link, split = host.run_view.how_link, host.results_split
            assert link.isVisibleTo(host), "a finished run left no 'How the run went'"
            gaps.append((split.geometry().top() - link.geometry().bottom() - 1,
                         design.scaled(design.GAP["head_table"])))
    assert gaps == [(6, 6), (9, 9)], gaps


def test_the_run_page_shows_what_the_model_says(tmp_path, monkeypatch):
    """A training's lines and rows, fed through the stand-in run with the window's clock moved by hand, are drawn
    on its page as its model reads them, at each moment: the sentence, the stages with their markers and words, the
    time axis's four moments and the line under it, the line that names what follows, the health table's rows and
    the messages, newest first, each opening with its marker's word and its date; each part read only while it is in
    sight. The health table's cells are drawn as the look draws a value and a state: a figure's number apart from
    what follows it, words alone plain, a state's marker beside its word."""
    from PySide6.QtCore import Qt
    from core.gui.widgets.rule_table import StateCell, TextCell, ValueCell
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    app = qt_app()
    monkeypatch.setenv("PRISM_VRAM_CEILING_GIB", "6")
    clock = Clock(_at(8, 19, 2, 11))
    held, host, stand_in = _controller_on(tmp_path, clock), build_run_host(), StandIn()
    view, text = host.run_view, RunText(held)
    start_run(host, stand_in, spec=_training())

    def said(line, level="info"):
        stand_in.say(line, level)
        turn_until(app, lambda: line in text.texts())

    def framed(done):
        stand_in.row(SIMULATION_ROW, done, 10_000)
        turn_until(app, lambda: held.model.batches_done() == done)

    said(_LINES["budget"])
    said(_LINES["rotation_begins"])
    clock.advance(90)
    said(_LINES["rotation_done"])
    clock.advance(9)
    said(_LINES["simulation_writing"])
    said(_LINES["mem"])
    framed(0)
    assert _page_says(view) == _model_says(held.model, clock(), "No end time yet: the first batches are being timed.")
    clock.advance(194)
    framed(20)
    said(_LINES["patho_batch"], "warning")
    clock.advance(60)
    said("a warning the window has no words of its own for", "warning")
    axis = held.model.axis()
    assert axis.end_from == axis.end_by is not None, axis
    assert _page_says(view) == _model_says(held.model, clock(),
                                           f"Expected to end by {texts.date_text(axis.end_by, axis.now)}")
    clock.advance(600)
    said(_LINES["simulation_complete"])
    for epoch in (1, 2):
        clock.advance(3780)
        said(_epoch(epoch))
    page = _page_says(view)
    assert page == _model_says(held.model, clock(), held.model.axis().unknown)
    assert [stage[1] for stage in page["stages"]] == ["done", "done", "now", "later"], page["stages"]
    assert len(page["health"]) == 3 and len(page["messages"]) == 2, page
    table = view.health.view.model()
    cells = [[table.index(row, column).data(Qt.UserRole) for column in range(3)] for row in range(table.rowCount())]
    assert cells == [                       # a number set apart from what follows it; a state's marker beside its word
        [TextCell("Failed simulations", wrap=True), ValueCell("0.15", "%"), StateCell("fine", "Fine")],
        [TextCell("Unusable driven recordings", wrap=True), ValueCell("", "Counted when simulation ends"),
         TextCell("")],
        [TextCell("Graphics-card memory", wrap=True), ValueCell("10.4", "of 15.9 GiB, limit 6 GiB, during simulation"),
         StateCell("fine", "Fine")]], cells
    stand_in.release()
    turn_until(app, lambda: not held.is_live())


def test_the_full_text_is_hidden_until_asked_and_follows_the_end_only_while_at_the_end(tmp_path):
    """The run's full text is hidden until "Show the full text", and "Hide the full text" hides it again. It holds
    one run's lines, the run's controller handing it each line as it says it, and the newest 5,000 of them,
    headed "Earlier lines are in the file." once older lines have gone. While it stands at its end it follows the
    end, also through lines that came while it was hidden; scrolled up, a line that arrives moves nothing, and
    lines that go from the top leave the line in sight where it was, the first time, when the head comes above
    them, as later."""
    from core.gui.widgets.log_pane import LINES_KEPT
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host = build_run_host()
    shown_off_screen(host, 1000, 800)
    view, full = host.run_view, host.run_view.full_text
    bar = full.verticalScrollBar()
    first = StandIn()
    start_run(host, first, spec=_training())
    first.say("a line of the first run")
    turn_until(app, lambda: "a line of the first run" in full.lines())
    first.release()
    turn_until(app, lambda: not held.is_live())

    second = StandIn()
    start_run(host, second, spec=_training(name="retrain_4"))
    count = 0

    def lines(n):
        nonlocal count
        held.lines.emit([(f"line {count + i}", "info") for i in range(n)])
        count += n
        app.processEvents()

    lines(60)                                           # while the full text is hidden
    assert (view.text_block.isVisibleTo(view), view.text_button.text()) == (False, "Show the full text")
    view.text_button.click()
    app.processEvents()
    assert (view.text_block.isVisibleTo(view), view.text_button.text()) == (True, "Hide the full text")
    assert "a line of the first run" not in full.lines() and full.lines()[0].startswith("Training retrain_4, run ")
    assert bar.maximum() > 0 and bar.value() == bar.maximum(), "the full text was not shown at its end"
    lines(10)
    assert bar.value() == bar.maximum(), "the full text did not follow the end"
    bar.setValue(5)
    top = full.firstVisibleBlock().text()
    lines(10)
    assert (bar.value(), full.firstVisibleBlock().text()) == (5, top), "a line that arrived moved the text"
    bar.setValue(bar.maximum())
    for _ in range(49):
        lines(100)
        assert bar.value() == bar.maximum(), count
    assert not full.is_cut()
    bar.setValue(bar.maximum() // 2)                    # scrolled up when the first lines go and the head comes
    top = full.firstVisibleBlock().text()
    lines(100)
    assert (full.is_cut(), full.firstVisibleBlock().text()) == (True, top), "the head's arrival moved the text"
    bar.setValue(bar.maximum())
    for _ in range(10):
        lines(100)
        assert bar.value() == bar.maximum(), count
    shown = full.toPlainText().split("\n")
    assert (shown[0], len(full.lines()), full.lines()[0], full.lines()[-1]) == (
        "Earlier lines are in the file.", LINES_KEPT, "line 1080", "line 6079"), (shown[:2], len(shown))
    bar.setValue(bar.maximum() // 2)
    top = full.firstVisibleBlock().text()
    lines(300)
    assert full.firstVisibleBlock().text() == top and bar.value() < bar.maximum(), "lines that went moved the text"
    assert len(full.lines()) == LINES_KEPT and full.lines()[-1] == "line 6379"
    bar.setValue(bar.maximum())                         # at its end again: it follows
    view.text_button.click()
    assert (view.text_block.isVisibleTo(view), view.text_button.text()) == (False, "Show the full text")
    lines(40)                                           # while it is hidden again
    view.text_button.click()
    app.processEvents()
    assert bar.value() == bar.maximum() and full.lines()[-1] == "line 6419", "the full text came back short of its end"
    second.release()
    turn_until(app, lambda: not held.is_live())


def test_the_full_text_finds_and_opens_its_file_and_its_folder(tmp_path, monkeypatch):
    """"Find…" opens a field that selects, at each Enter, the next place in the full text that holds its words,
    from the top again after the end, and leaves the selection where it was when no line holds them. "Open the
    file" and "Open its folder" hand the run's text file and the folder of runs to the computer, which opens
    them; no path is written anywhere on the page. A run that keeps no file has neither to open."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, running, start_run, turn_until
    app = qt_app()
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host, stand_in = build_run_host(), StandIn()
    view, full = host.run_view, host.run_view.full_text
    start_run(host, stand_in, spec=_training())
    for line in ("alpha one", "beta one", "alpha two"):
        stand_in.say(line)
    turn_until(app, lambda: "alpha two" in full.lines())
    view.text_button.click()
    assert not view.find_field.isVisibleTo(view)
    view.find_button.click()
    assert view.find_field.isVisibleTo(view)
    found = []

    def find(words):
        view.find_field.setText(words)
        view.find_field.returnPressed.emit()
        cursor = full.textCursor()
        found.append((cursor.selectedText(), cursor.block().text()))

    for words in ("alpha", "alpha", "alpha", "gamma"):
        find(words)
    assert found == [("alpha", "alpha one"), ("alpha", "alpha two"), ("alpha", "alpha one"), ("alpha", "alpha one")]
    view.file_button.click()
    view.folder_button.click()
    folder, run = tmp_path / "runs", "20261008T190211-training"
    assert opened == [QUrl.fromLocalFile(str(folder / f"{run}.txt")), QUrl.fromLocalFile(str(folder))], opened
    words = ([widget.text() for widget in view.findChildren(QLabel) + view.findChildren(QAbstractButton)
              + view.findChildren(QLineEdit)] + [full.toPlainText(), view.title.text()])
    assert not [said for said in words if str(tmp_path) in said or tmp_path.as_posix() in said], words
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    with running(host, spec=RunSpec("Run", None, keeps_files=False)):
        assert not view.file_button.isEnabled() and not view.folder_button.isEnabled()


def test_live_simulation_keeps_its_own_view_under_the_runs_sentence(tmp_path, monkeypatch):
    """A live simulation, a run with a view of its own, keeps that view where it is: the run's page stands above it
    in its compact form, the sentence and the stop button alone, and its sentence counts the samples shown against
    the samples to come, as the page hands them over with each frame; a stage the run says it begins is named
    there. The stop button stands at the right of the page at its own width. At its end the page above the view
    goes. When the samples to come cannot be worked out at the start, the run starts all the same, counting
    none, and meets the failure itself."""
    import numpy as np
    from core.gui.panels import simulate_panel as live
    from tests._fixtures import all_asked, build_part, forget_asked, qt_app
    from tests._runs import Clock, StandIn, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    cell = tmp_path / "cell.txt"
    cell.write_text("# values\n", encoding="utf-8")
    stand_in = StandIn(word="Live simulation")
    monkeypatch.setattr(live, "build_stream_config", lambda model, path: SimpleNamespace(length_unit="nm"))
    monkeypatch.setattr(live, "plan_stream", lambda cfg, t_obs: SimpleNamespace(total_steps=7_999,
                                                                                 subsample_factor=2))
    monkeypatch.setattr(live, "run_simulation_stream", stand_in)
    panel = build_part(live.SimulatePanel)
    panel.cell_picker.selected_path = lambda: str(cell)
    shown_off_screen(panel, 1300, 900)
    view = panel.run_view
    assert not view.isVisibleTo(panel)
    panel._start()
    app.processEvents()
    assert held.is_live() and held.model.spec.own_view and view.compact
    assert view.isVisibleTo(panel) and panel.results.currentWidget() is panel.figures
    assert panel.results_split.indexOf(panel.live_view) == 0 and panel.live_view.isVisibleTo(panel)
    shown = [name for name in ("title", "text_button", "stop_button", "back_button", "stage_line", "sentence", "axis",
                               "axis_caption", "follows", "body", "text_block")
             if getattr(view, name).isVisibleTo(view)]
    assert shown == ["stop_button", "sentence"], shown
    assert view.stop_button.text() == "Stop simulation"
    stop = view.stop_button.geometry()
    assert (stop.width(), stop.right() + 1) == (view.stop_button.sizeHint().width(), view.head.width()), (
        "the compact page's stop button does not stand at the right at its own width", stop, view.head.width())
    panel._on_chunk(np.zeros((1_200, 2)))
    assert view.sentence.text() == "Live simulation is running: 1,200 of 4,000 samples."
    view.set_stage("Recording")                        # a run that says its stages itself
    assert (held.model.stage, view.sentence.text()) == ("Recording", "Live simulation is running: Recording.")
    view.stop_button.click()
    turn_until(app, lambda: not held.is_live())
    assert not view.isVisibleTo(panel) and panel.live_view.isVisibleTo(panel)

    def unplanned(cfg, t_obs):
        raise RuntimeError("the samples to come cannot be worked out")

    again = StandIn(word="Live simulation")
    monkeypatch.setattr(live, "plan_stream", unplanned)
    monkeypatch.setattr(live, "run_simulation_stream", again)
    forget_asked()
    panel._start()
    assert (held.is_live(), all_asked()) == (True, []), (
        "a count that could not be worked out kept the run from starting")
    panel._on_chunk(np.zeros((1_200, 2)))
    assert view.sentence.text() == "Live simulation is starting.", view.sentence.text()
    again.release()
    turn_until(app, lambda: not held.is_live())


def test_when_a_run_starts_from_the_keyboard_the_keyboard_goes_to_its_stop_button(tmp_path, monkeypatch):
    """A run started from a page's form with a key leaves the keyboard on the run page's stop button, where the
    form's lock would leave it on nothing; it is given for no key of its own, so its ring shows while the keyboard
    has the turn. A run started while the keyboard is elsewhere leaves it where it was. A load started with a key,
    whose page never takes the column, gives the keyboard to no stop button out of sight."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    from tests._window_helpers import given, in_front, shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host = build_run_host()
    start = QPushButton("Start the run")
    host.controls_layout.addWidget(start)
    first = StandIn()
    start.clicked.connect(lambda: start_run(host, first, spec=_training()))
    shown_off_screen(host, 1000, 700)
    focus = given(monkeypatch)
    with in_front(host):
        start.setFocus(Qt.TabFocusReason)
        QTest.keyClick(start, Qt.Key_Space)
        assert held.is_live() and not start.isEnabled()
        stop = host.run_view.stop_button
        assert (QApplication.focusWidget(), focus[-1]) == (stop, (stop, Qt.OtherFocusReason)), (
            "the keyboard did not go to the run's stop button")
        first.release()
        turn_until(app, lambda: not held.is_live())
        elsewhere = build_run_host()
        field = QLineEdit()
        elsewhere.figures_layout.addWidget(field)          # beside its figures: no run locks it
        shown_off_screen(elsewhere)
        with in_front(elsewhere):
            field.setFocus(Qt.TabFocusReason)
            given_before, second = len(focus), StandIn()
            start_run(host, second, spec=_training())
            assert QApplication.focusWidget() is field and focus[given_before:] == [], focus[given_before:]
            second.release()
            turn_until(app, lambda: not held.is_live())
    loads, load, third = build_run_host(), QPushButton("Load the record"), StandIn(word="Prior load")
    loads.controls_layout.addWidget(load)
    load.clicked.connect(lambda: start_run(loads, third, spec=_load()))
    shown_off_screen(loads, 1000, 700)
    with in_front(loads):
        load.setFocus(Qt.TabFocusReason)
        given_before = len(focus)
        QTest.keyClick(load, Qt.Key_Space)
        assert held.is_live() and loads.results.currentWidget() is loads.figures
        assert focus[given_before:] == [], "a load's start gave the keyboard to a stop button out of sight"
        third.release()
        turn_until(app, lambda: not held.is_live())


def test_at_a_runs_end_the_keyboard_its_page_held_goes_to_a_named_place(tmp_path, monkeypatch):
    """At a run's end a keyboard its page held goes to a named place, given for no key of its own so that its ring
    shows while the keyboard has the turn: "How the run went" after a finish, which gives the column back to the
    figures, and "Back to the form" after a failure, whose page stays. A keyboard the page did not hold stays where
    it was."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton
    from tests._fixtures import build_run_host, qt_app
    from tests._runs import Clock, StandIn, start_run, turn_until
    from tests._window_helpers import given, in_front, shown_off_screen
    app = qt_app()
    held = _controller_on(tmp_path, Clock(_at(8, 19, 2, 11)))
    host, start, runs = build_run_host(), QPushButton("Start the run"), [StandIn(), StandIn()]
    host.controls_layout.addWidget(start)
    start.clicked.connect(lambda: start_run(host, runs[0], spec=_training()))
    shown_off_screen(host, 1000, 700)
    view = host.run_view
    focus = given(monkeypatch)
    with in_front(host):
        for end, place in ((StandIn.release, view.how_link),
                           (lambda run: run.fail(RuntimeError("the run broke")), view.back_button)):
            start.setFocus(Qt.TabFocusReason)
            QTest.keyClick(start, Qt.Key_Space)
            assert QApplication.focusWidget() is view.stop_button
            end(runs.pop(0))
            turn_until(app, lambda: not held.is_live())
            assert (QApplication.focusWidget(), focus[-1]) == (place, (place, Qt.OtherFocusReason)), (
                f"at the run's end the keyboard did not go to '{place.text()}'")
        view.back_button.click()
        elsewhere, field = build_run_host(), QLineEdit()
        elsewhere.figures_layout.addWidget(field)          # beside its figures: no run locks it
        shown_off_screen(elsewhere)
        with in_front(elsewhere):
            field.setFocus(Qt.TabFocusReason)
            given_before, third = len(focus), StandIn()
            start_run(host, third, spec=_training())
            third.release()
            turn_until(app, lambda: not held.is_live())
            assert QApplication.focusWidget() is field and focus[given_before:] == [], focus[given_before:]


def test_the_run_page_draws_with_the_looks_devices_at_every_scaling(tmp_path):
    """The run's page draws with the look's devices, in both themes and at the four display scalings: its health
    table between three rules, each a whole number of device pixels (2, 1, 2 at 100 % and 125 %, 3, 1, 3 at 150 %,
    4, 2, 4 at 200 %); no pixel in the accent but in the marker of the stage in hand and on the time axis, the two
    places the look gives the present; the sentence in the lettering face at the heading's size and at weight 400,
    read from its font; the full text in the fixed-width face. Drawn as the look's suite draws, the page shown
    where no screen sees it and rendered into a picture at each scaling; no assertion reads the pixels of
    lettering. Laid out: the head's title at its left takes the room its buttons leave, each button at its own
    width and the stop button at the right; the stages stand on one line, two to a line when the page is too
    narrow for them, and on one line again when it is wide."""
    import array
    import math
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QFont
    from core.gui import design, fonts
    from core.gui.widgets.marker import Marker
    from tests._fixtures import build_run_host, qt_app
    from tests._look_helpers import _screen_of, _settle, _the_look
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host, stand_in = _controller_on(tmp_path, clock), build_run_host(), StandIn()
    view, text = host.run_view, RunText(held)
    start_run(host, stand_in, spec=_training())
    for line, level in ((_LINES["budget"], "info"), (_LINES["rotation_begins"], "info"),
                        (_LINES["rotation_done"], "info"), (_LINES["simulation_writing"], "info"),
                        (_LINES["mem"], "info"), (_LINES["patho_batch"], "warning")):
        stand_in.say(line, level)
    stand_in.row(SIMULATION_ROW, 20, 10_000)
    turn_until(app, lambda: held.model.batches_done() == 20 and _LINES["patho_batch"] in text.texts())
    shown_off_screen(host, 1300, 1000)
    _settle()
    now_marker, = [shown for shown in view.stage_line.findChildren(Marker) if shown.kind == "now"]
    rules = {1.0: [2, 1, 2], 1.25: [2, 1, 2], 1.5: [3, 1, 3], 2.0: [4, 2, 4]}
    for dark in (False, True):
        with _the_look(dark):
            _settle()
            font = view.sentence.font()
            assert (font.family(), font.pixelSize(), font.weight()) == (
                design.LETTERING, design.scaled(design.TYPE["head"][0]), QFont.Weight(400)), font.toString()
            assert view.full_text.font().families() == fonts.fixed().families()
            for ratio, want in rules.items():
                image = _screen_of(view, ratio)
                table = view.health.mapTo(view, QPoint(0, 0))
                x = round((table.x() + 3) * ratio)
                rows = range(math.floor(table.y() * ratio), math.ceil((table.y() + view.health.height()) * ratio))
                column = [image.pixelColor(x, y).rgb() == design.colour("rule").rgb() for y in rows]
                runs = [len(run) for run in "".join("1" if on else "0" for on in column).split("0") if run]
                assert runs == want, (dark, ratio, runs)
                allowed = [QRect(widget.mapTo(view, QPoint(0, 0)), widget.size()) for widget in (now_marker, view.axis)]
                pixels = array.array("I")
                pixels.frombytes(bytes(image.constBits()))
                across, accent = image.bytesPerLine() // 4, design.colour("accent").rgba()
                stray = [(at % across, at // across) for at, pixel in enumerate(pixels) if pixel == accent
                         and not any(box.contains(QPoint(int((at % across) / ratio), int((at // across) / ratio)))
                                     for box in allowed)]
                assert stray == [], (dark, ratio, stray[:5])
            title, show, stop = view.title.geometry(), view.text_button.geometry(), view.stop_button.geometry()
            gap = design.scaled(design.GAP["head_buttons"])
            assert (title.left(), title.right() + 1 + gap, show.right() + 1 + gap, stop.right() + 1, show.width(),
                    stop.width()) == (0, show.left(), stop.left(), view.head.width(),
                                      view.text_button.sizeHint().width(), view.stop_button.sizeHint().width()), (
                dark, title, show, stop, view.head.width())

    def places():
        grid = view.stage_line._grid
        return [grid.getItemPosition(grid.indexOf(holder))[:2] for holder, *_parts in view.stage_line._items]

    with _the_look(False):
        _settle()
        assert places() == [(0, 0), (0, 1), (0, 2), (0, 3)], places()
        host.resize(700, 1000)                          # too narrow for the four stages on one line
        _settle()
        assert places() == [(0, 0), (0, 1), (1, 0), (1, 1)], places()
        host.resize(1300, 1000)
        _settle()
        assert places() == [(0, 0), (0, 1), (0, 2), (0, 3)], places()
    stand_in.release()
    turn_until(app, lambda: not held.is_live())


def test_the_run_page_shows_the_loss_curve_during_the_fit_and_after_it(tmp_path):
    """The fit's loss by epoch stands on the run's page under the training's health from the fit's first record on:
    not before the fit, nor in the fit before its first record. Each record is drawn as its line arrives, the best
    epoch marked as the model counts it, and the caption names the run's patience. It stays when the run has ended,
    and on the page drawn again from the run's two files, where the patience is not known and the caption leaves out
    when the fit ends. It stands between the health table and the messages, the look's gap between blocks before it
    and after it; with no curve the messages stand that gap under the health table. The page's next run starts with
    no curve."""
    from PySide6.QtCore import QPoint
    from core.gui import design
    from core.gui.run import history
    from tests._fixtures import build_run_host, qt_app
    from tests._look_helpers import _settle
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host, stand_in = _controller_on(tmp_path, clock), build_run_host(), StandIn()
    view, text = host.run_view, RunText(held)
    start_run(host, stand_in, spec=_training())
    shown_off_screen(host, 1300, 1000)

    def said(line, level="info"):
        stand_in.say(line, level)
        turn_until(app, lambda: line in text.texts())

    def curve():
        """Whether the curve is in the page, the epochs it draws, and what its accessible description says."""
        return view.loss.isVisibleTo(view), view.loss.drawn()["epochs"], view.loss.accessibleDescription()

    def spans():
        """Where the health table, the curve and the messages' head stand down the page: each one's top and bottom."""
        _settle()
        return [(part.mapTo(view, QPoint(0, 0)).y(), part.mapTo(view, QPoint(0, part.height())).y())
                for part in (view.health, view.loss, view.messages_head)]

    assert curve() == (False, [], ""), "the curve stood at the start"
    said(_LINES["budget"])
    said(_LINES["simulation_complete"])
    said(_LINES["patho_batch"], "warning")                    # a message, so that the messages stand under the curve
    assert (held.model.stage, curve()) == ("Fitting", (False, [], "")), "the curve stood before the fit's first record"
    block = design.scaled(design.GAP["block"])
    health, _hidden, head = spans()
    assert head[0] - health[1] == block, ("with no curve the messages do not stand a block's gap under the health "
                                          "table", health, head)
    for epoch, since, validation in ((1, 0, -3.0), (2, 0, -3.4), (3, 1, -3.2)):
        clock.advance(3780)
        said(_epoch(epoch, since=since, validation=validation))
        assert curve()[:2] == (True, list(range(1, epoch + 1))), epoch
    assert held.model.best_epoch() == 2 and curve()[2] == "epochs 1 to 3; best so far: epoch 2"
    health, fit, head = spans()
    assert view.messages_head.isVisibleTo(view) and (fit[0] - health[1], head[0] - fit[1]) == (block, block), (
        health, fit, head)
    assert view.loss.caption.plain_text().endswith("Fitting ends when validation has not improved for 20 epochs.")
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
    view.how_link.click()
    assert (view.mode, curve()) == ("ended", (True, [1, 2, 3], "epochs 1 to 3; best so far: epoch 2"))
    run = held.status.run
    view.show_past(history.read(tmp_path / "runs" / f"{run}.json"), tmp_path / "runs" / f"{run}.txt")
    assert (view.mode, curve()) == ("past", (True, [1, 2, 3], "epochs 1 to 3; best so far: epoch 2"))
    assert view.loss.caption.plain_text() == "Loss by epoch. Loss is the negative log-probability; lower is better."
    view.begin(_training())                                  # a new run of the page, before it is first drawn
    assert curve() == (False, [], ""), "a new run of the page kept the last run's curve"
    second = StandIn()
    start_run(host, second, spec=_training())
    assert (view.mode, curve()) == ("live", (False, [], "")), "the page's next run kept the last run's curve"
    second.release()
    turn_until(app, lambda: not held.is_live())


def test_the_loss_curve_never_widens_the_run_pages_column(tmp_path):
    """At the window's least size the run's page is 326 px wide, narrower than the loss curve's least width. The
    curve does not widen the page's scrolling column: before the fit's first record and after three, the column is
    as wide as the room it scrolls in, and the health table, the messages and the curve end inside it. The plot keeps
    its least width there and is cut at its right."""
    from PySide6.QtCore import QPoint
    from core.gui import design
    from tests._fixtures import build_run_host, qt_app
    from tests._look_helpers import _settle
    from tests._runs import Clock, RunText, StandIn, start_run, turn_until
    from tests._window_helpers import shown_off_screen
    app = qt_app()
    clock = Clock(_at(8, 19, 2, 11))
    held, host, stand_in = _controller_on(tmp_path, clock), build_run_host(), StandIn()
    view, text = host.run_view, RunText(held)
    start_run(host, stand_in, spec=_training())
    view.setFixedWidth(326)                                   # the run's page at the window's least size
    shown_off_screen(host, 1300, 1000)
    for line, level in ((_LINES["budget"], "info"), (_LINES["simulation_complete"], "info"),
                        (_LINES["patho_batch"], "warning")):
        stand_in.say(line, level)
    turn_until(app, lambda: _LINES["patho_batch"] in text.texts())

    def widths():
        """The room the column scrolls in, the column's width, and where the health table, the messages and the
        curve end across it, each while in sight."""
        _settle()
        column = view.body.widget()
        ends = [part.mapTo(column, QPoint(part.width(), 0)).x() if part.isVisibleTo(view) else None
                for part in (view.health, view.messages, view.loss)]
        return view.body.viewport().width(), column.width(), ends

    room, wide, (health, messages, curve) = widths()
    assert (wide, curve) == (room, None) and None not in (health, messages), (room, wide, health, messages, curve)
    assert max(health, messages) <= room, (room, health, messages)
    for epoch in (1, 2, 3):
        clock.advance(3780)
        stand_in.say(_epoch(epoch, validation=-3.1 - epoch / 10))
    turn_until(app, lambda: len(held.model.epochs()) == 3)
    room, wide, (health, messages, curve) = widths()
    assert wide == room, ("the loss curve widened the run page's column", room, wide)
    assert None not in (health, messages, curve), (health, messages, curve)
    assert max(health, messages, curve) <= room, (room, health, messages, curve)
    least = design.scaled(design.FIGURE["loss"]["min_w"])
    assert view.loss.plot.width() == least > view.loss.width(), (least, view.loss.plot.width(), view.loss.width())
    stand_in.release()
    turn_until(app, lambda: not held.is_live())
