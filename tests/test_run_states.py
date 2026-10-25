"""The run's model and the table of the engine's lines it reads, with no window and no thread: a run's lines and
progress rows are fed to it, and the window's clock is handed in, so a run of days is read in a moment.

A progress row is the window's own parse of a frame of the engine's progress bar; an epoch's record is the window's
literal copy of the line the training call writes, which the first test holds to the engine's own function.
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
    """Three more copies of the engine's words, read by the model outside its table, are held to the engine as its
    rows are, off the syntax tree of the module that writes them: the name of a training batch inside a line, from
    which four rows read their batch; the description of the progress row that counts a training's simulation; and
    the training library's counter of the epochs it has trained, the fit's only sign when the library writes no
    record per epoch (read from the library's installed source)."""
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
