"""The run's text and its history file.

Every finished line a run writes reaches a dated text file as it is written, whichever of the three channels it
came by (a print, a record, a Python warning); the file's writer can never raise into a run, speak, or take a
requested stop inside a save; the worker writes how a run ended before it says anything to the window; a library's
record reaches a run at its own level; and each run keeps one history file beside the settings file.

No window is built here and no test sleeps: a run is a worker run on the test's own thread, every moment is read
off a clock the test holds, and a child process is waited for by reading what it prints.

Run: pytest tests/test_run_text.py
"""
import json
import logging
import os
import subprocess
import sys
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from core import runs
from core.gui import streams
from core.gui.run import textfile
from core.gui.run.textfile import RunTextFile
from core.gui.worker import CANCEL_TOO_LATE_TEXT, Worker, WorkerSignals
from tests._fixtures import calls_that_speak, code_only, qt_app

REPO = Path(__file__).resolve().parents[1]
# The moment the lines of these tests are stamped with, in local time, and its stamp.
AT = datetime(2026, 10, 8, 19, 2, 11).astimezone()
STAMP = "2026-10-08 19:02:11"
INDENT = " " * 20


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, start=AT):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


def _lines(path):
    """The lines of a text file as a reader sees them; none for a file that is not there."""
    return Path(path).read_text(encoding="utf-8").splitlines() if Path(path).exists() else []


def _run(fn, run_text, *, cancel=None, **slots):
    """Run ``fn`` through a worker on this thread, ``run_text`` its text file, each of ``slots`` (``log=``,
    ``error=``, ``cancelled=``, ``result=``, ``finished=``) connected to that signal first."""
    worker = Worker(fn, cancel=cancel, run_text=run_text)
    for name, slot in slots.items():
        getattr(worker.signals, name).connect(slot)
    worker.run()
    return worker


# ── the text file ────────────────────────────────────────────────────────────────────────────────
def test_every_finished_line_of_the_three_channels_reaches_the_file_as_it_happens(tmp_path):
    """A print, a ``core`` record at warning, a Python warning, a print on the error stream and a library's warning
    record each reach the run's text file in the order they were written, at their own levels: the library's PRINT
    at ``err`` (no warning mark), its warning RECORD at warning. A line is in the file as soon as it is written: the
    run reads its file right after its own first print and finds the line there. The worker's own last line follows."""
    from core import logging_root
    from core.gui.app import _library_record_sink

    qt_app()
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    seen = []

    def run():
        print("Simulating batch 1 of 4")
        seen.append(_lines(text.path))
        logging.getLogger("core.probe").warning("[checkpoint] the set is saved every 50 batches")
        warnings.warn("a setting the run cannot honour", UserWarning)
        print("a library's own line on the error stream", file=sys.stderr)
        logging.getLogger("sbi.inference").warning("Only 0.5% proposal samples are accepted.")
        return "done"

    logging_root.install(_library_record_sink)
    try:
        _run(run, text)
    finally:
        logging_root.remove()
        text.close()
    assert seen == [[f"{STAMP} info Simulating batch 1 of 4"]], seen
    assert _lines(text.path) == [
        f"{STAMP} info Simulating batch 1 of 4",
        f"{STAMP} warning [checkpoint] the set is saved every 50 batches",
        f"{STAMP} warning UserWarning: a setting the run cannot honour",
        f"{STAMP} err a library's own line on the error stream",
        f"{STAMP} warning library: sbi.inference: Only 0.5% proposal samples are accepted.",
        f"{STAMP} window The run finished."], _lines(text.path)


def test_the_line_format_names_the_date_the_level_and_indents_a_traceback(tmp_path):
    """A line is the clock's local date and time, the level and the text, one space between each. The head names
    the run, its name left out when it has none, and the memory limit in force, left out when none is handed; a try
    after the first opens with a line of its own; the further lines of a text written whole are indented by the
    stamp's width, under its text. The file is UTF-8 whatever the machine's own encoding, so a line of the reduction
    map's symbols reads back as it was written, and a character no encoding can hold (a lone surrogate a library let
    through) is written escaped instead of breaking the file. Once closed, the writer writes nothing more."""
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    text.head(word="Training", name="retrain_3", run="20261008T190211-training",
              memory_limit="The graphics card's memory limit in force is 6 GiB, set in Settings.")
    text.separator(2)
    text.line('Traceback (most recent call last):\n  File "pipeline.py", line 12, in run\nRuntimeError: the batch '
              'failed', "error")
    text.line("ξ* = 0.5 χ", "info")
    text.line("a library's stray \udcff", "err")
    text.close()
    closed = text.path.read_bytes()
    text.line("a line after the end", "info")
    assert text.path.read_bytes() == closed, "the writer wrote after it was closed"
    unnamed = RunTextFile(tmp_path / "prior.txt", clock=_Clock())
    unnamed.head(word="Prior", name="", run="20261008T190211-prior", memory_limit="")
    unnamed.close()
    assert textfile.STAMP_WIDTH == len(f"{STAMP} ") == len(INDENT)
    assert textfile.LEVELS == ("info", "warning", "error", "err", "window")
    assert _lines(text.path) == [
        f"{STAMP} window Training retrain_3, run 20261008T190211-training.",
        f"{STAMP} window The graphics card's memory limit in force is 6 GiB, set in Settings.",
        f"{STAMP} window Try 2 begins here.",
        f"{STAMP} error Traceback (most recent call last):",
        f'{INDENT}  File "pipeline.py", line 12, in run',
        f"{INDENT}RuntimeError: the batch failed",
        f"{STAMP} info ξ* = 0.5 χ",
        f"{STAMP} err a library's stray \\udcff"], _lines(text.path)
    assert _lines(unnamed.path) == [f"{STAMP} window Prior, run 20261008T190211-prior."], _lines(unnamed.path)


_WRITE_THEN_WAIT = """
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from core.gui.run.textfile import RunTextFile
text = RunTextFile(Path(sys.argv[1]), clock=lambda: datetime(2026, 10, 8, 19, 2, 11).astimezone())
for n in (1, 2, 3):
    text.line(f"line {n} of the run", "info")
print("written", sorted(name for name in ("torch", "PySide6") if name in sys.modules), flush=True)
sys.stdin.read()
"""


def test_the_file_survives_a_kill_of_the_process_writing_it(tmp_path):
    """Every line is handed to the operating system as it is written, so a process killed outright, which cleans up
    nothing of its own, leaves every finished line in the file. The child writes three lines, says so and waits on
    its input; it is killed once it has said so, never after a timed wait. The writer loads neither torch nor the
    toolkit: the child imports it alone and says what it loaded."""
    path = tmp_path / "run.txt"
    with subprocess.Popen([sys.executable, "-c", _WRITE_THEN_WAIT, str(path), str(REPO)], cwd=REPO,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        guard = threading.Timer(120, child.kill)        # a child that hangs is ended, and the test fails
        guard.start()
        try:
            said = child.stdout.readline().strip()
            child.kill()
            child.wait()
        finally:
            guard.cancel()
        trouble = child.stderr.read()
    assert said == "written []", (said, trouble)
    assert _lines(path) == [f"{STAMP} info line {n} of the run" for n in (1, 2, 3)], _lines(path)


class _NoRoom:
    """A file every write to which fails, as a full disk's does."""

    def write(self, text):
        raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def fileno(self):
        raise OSError(9, "Bad file descriptor")

    def close(self):
        pass


class _Heard(logging.Handler):
    """Every record a logger passes on, its words kept."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.words = []

    def emit(self, record):
        self.words.append(record.getMessage())


def _every_kind_of_line(writer):
    writer.head(word="Training", name="retrain_3", run="20261008T190211-training",
                memory_limit="The graphics card's memory limit in force is 6 GiB, set in Settings.")
    writer.separator(2)
    writer.line("[checkpoint] writing batches 0 to 50 of 10,000", "info")
    writer.outcome("The run failed: OSError: the disk is full", "error", "Traceback (most recent call last):\n")
    writer.sync()


def test_the_text_files_writer_never_raises_into_a_run_and_says_nothing_inside_a_save(tmp_path, monkeypatch):
    """The run's line sink is called on the thread that wrote the line, under the stream pump's lock, and
    the engine writes lines inside a save that must not be interrupted (the simulation set's commit, a pass of the
    fit being saved). So the writer never raises into the run, never prints, never logs, and is never a point where
    a requested stop is taken: a raise there would stop the run halfway through a save, and a line it printed or
    logged would come back into the same pump, under the lock it already holds.

    A file that cannot be written, reached through the pump as a run reaches it: the run goes on, the writer is
    broken, the pump queues the window's one sentence saying so at warning, and the file is not tried again. Then,
    inside a live redirect whose stop has been requested, a working writer and one that breaks write every kind of
    line, first inside a save's section and then outside it: nothing is raised, nothing reaches the run's streams or
    the ``core`` logger, and the stop is never taken. Last, the writer's code holds no call that prints, logs or
    writes to a standard stream."""
    qt_app()
    real_open, opened = open, []

    def opener(path, *args, **kwargs):
        if Path(path).name.startswith("no-room"):
            opened.append(Path(path).name)
            return _NoRoom()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(textfile, "open", opener, raising=False)

    broken = RunTextFile(tmp_path / "no-room.txt", clock=_Clock())
    batches = []
    pump = streams._Pump(SimpleNamespace(log_batch=SimpleNamespace(emit=batches.append),
                                         rows=SimpleNamespace(emit=lambda rows: None)), line_sink=broken.line)
    pump.sink("log", ("Simulating batch 1 of 4", "info"))
    pump.sink("log", ("Simulating batch 2 of 4", "info"))
    pump.stop()
    assert broken.broken is True and opened == ["no-room.txt"], (broken.broken, opened)
    assert [line for batch in batches for line in batch] == [
        ("Simulating batch 1 of 4", "info"),
        ("The run's text file could not be written (OSError: [Errno 28] No space left on device); "
         "the full text is still on the page.", "warning"),
        ("Simulating batch 2 of 4", "info")], batches

    working = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    inside, outside = (RunTextFile(tmp_path / f"no-room-{n}.txt", clock=_Clock()) for n in (1, 2))
    heard, spoken, token, raised = _Heard(), [], streams.CancelToken(), []
    logging.getLogger("core").addHandler(heard)
    try:
        with streams.redirect_streams(WorkerSignals(), token, line_sink=lambda words, level: spoken.append(words)):
            token.requested.set()
            try:
                with runs.cancel_deferred():
                    for writer in (working, inside):
                        _every_kind_of_line(writer)
                for writer in (working, outside):
                    _every_kind_of_line(writer)
            except streams.WorkerCancelled:
                raised.append("a stop was taken")
            fired = token.fired
    finally:
        logging.getLogger("core").removeHandler(heard)
    for writer in (working, inside, outside):
        writer.close()
    assert (raised, fired) == ([], False), "the writer took the requested stop"
    assert (heard.words, spoken) == ([], []), (heard.words, spoken)
    assert (working.broken, inside.broken, outside.broken) == (False, True, True)
    assert len(_lines(working.path)) == 12, _lines(working.path)
    assert calls_that_speak(code_only(textfile)) == [], calls_that_speak(code_only(textfile))


def test_the_disk_is_synced_at_most_every_thirty_seconds_and_at_the_end(tmp_path, monkeypatch):
    """A line is handed to the operating system as it is written but never waits for the disk: only ``sync``, which
    the window calls from a timer, asks the disk to hold the file, at most every thirty seconds of the run's clock;
    ``close`` always does. Both ask the disk outside the writer's lock: a line takes that lock while the stream
    pump's lock is held, so a request made under it would stall the run's every print for the disk's time."""
    clock = _Clock()
    text = RunTextFile(tmp_path / "run.txt", clock=clock)
    synced = []                                   # for each request of the disk, whether the writer's lock was held
    monkeypatch.setattr(textfile.os, "fsync", lambda handle: synced.append(text._lock.locked()))
    for n in range(40):
        text.line(f"Simulating batch {n + 1} of 40", "info")
    assert synced == [], "a line waited for the disk"
    text.sync()
    assert len(synced) == 1, "the first sync did not reach the disk"
    clock.advance(29)
    text.sync()
    assert len(synced) == 1, "synced again after 29 seconds"
    clock.advance(1)
    text.sync()
    text.line("[checkpoint] writing batches 0 to 40 of 40", "info")
    assert len(synced) == 2, "not synced after 30 seconds"
    text.close()
    assert len(synced) == 3, "the end was not synced"
    assert synced == [False, False, False], "the disk was asked under the writer's lock"
    assert textfile.FSYNC_EVERY_S == 30.0
    assert len(_lines(text.path)) == 41


def test_a_sync_that_cannot_be_made_raises_nothing_and_is_made_again_when_due(tmp_path, monkeypatch):
    """``sync`` runs on the window's timer, not in the run, so nothing it meets may reach the window: a request that
    cannot be made (the file's handle cannot be copied) and one the disk refuses are each dropped without a word, the
    writer goes on writing, and the next request is made when it falls due."""
    clock = _Clock()
    text = RunTextFile(tmp_path / "run.txt", clock=clock)
    copied, asked, real_dup = [], [], os.dup

    def dup(handle):
        copied.append(handle)
        if len(copied) == 1:
            raise OSError(24, "Too many open files")
        return real_dup(handle)

    def fsync(handle):
        asked.append(handle)
        if len(asked) == 1:
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(textfile.os, "dup", dup)
    monkeypatch.setattr(textfile.os, "fsync", fsync)
    text.line("Simulating batch 1 of 4", "info")
    text.sync()                                             # the handle cannot be copied
    clock.advance(30)
    text.sync()                                             # the disk refuses
    assert (len(copied), len(asked)) == (2, 1), (copied, asked)
    text.line("Simulating batch 2 of 4", "info")
    clock.advance(30)
    text.sync()
    assert (len(copied), len(asked)) == (3, 2), "the request after the refused one was never made"
    text.close()
    assert text.broken is False
    assert _lines(text.path) == [f"{STAMP} info Simulating batch 1 of 4",
                                 "2026-10-08 19:02:41 info Simulating batch 2 of 4"], _lines(text.path)


def test_a_pump_whose_window_is_gone_keeps_no_more_lines_but_the_file_still_gets_them(tmp_path):
    """A run whose window has gone (its signals deleted while it goes on) stops the pump at the first signal it
    cannot emit: its thread ends by itself. From then on the pump keeps no line for the window: they would pile up
    unread for the rest of the run. The run's text file still gets every line the run writes."""
    import shiboken6

    qt_app()
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    signals = WorkerSignals()
    pump = streams._Pump(signals, line_sink=text.line)
    shiboken6.delete(signals)
    pump.sink("log", ("Simulating batch 1 of 4", "info"))
    pump._thread.join(timeout=60)                           # its next publish meets the deleted signals
    assert not pump._thread.is_alive(), "the pump went on after its window had gone"
    pump.sink("log", ("Simulating batch 2 of 4", "info"))
    pump.sink("log", ("[checkpoint] writing batches 0 to 2 of 4", "warning"))
    assert pump._logs == [], f"a pump with no window kept {pump._logs}"
    pump.stop()
    text.close()
    assert _lines(text.path) == [f"{STAMP} info Simulating batch 1 of 4", f"{STAMP} info Simulating batch 2 of 4",
                                 f"{STAMP} warning [checkpoint] writing batches 0 to 2 of 4"], _lines(text.path)


def test_the_worker_writes_its_outcome_before_it_says_anything(tmp_path):
    """How a run ended is in its text file before the worker says anything to the window: a slot on each signal
    reads the file as the signal arrives. A failure is its kind and its message, at error (its kind alone when it
    has no message, or a message that cannot be had: it is still said), its traceback indented under it; a stop and
    a finish are one line each, the window's own. The lines the worker says itself, outside the pump, are written
    too, each before it is said: the failure a stop may have replaced and its traceback, a stop that came too late
    to stop anything, and the redirect declined because another run holds the streams."""
    qt_app()

    def watched(fn, name, cancel=None):
        text = RunTextFile(tmp_path / f"{name}.txt", clock=_Clock())
        said = []

        def slot(signal):
            return lambda *args: said.append((signal, _lines(text.path)))

        _run(fn, text, cancel=cancel, **{signal: slot(signal) for signal in ("log", "error", "cancelled", "result")})
        text.close()
        return said

    def fails():
        raise ValueError("the batch failed")

    said = watched(fails, "failed")
    assert [signal for signal, _ in said] == ["error"], said
    held = said[0][1]
    assert held[0] == f"{STAMP} error The run failed: ValueError: the batch failed", held
    assert held[1] == f"{INDENT}Traceback (most recent call last):", held
    assert held[-1] == f"{INDENT}ValueError: the batch failed", held
    assert all(line.startswith(INDENT) for line in held[1:]), held

    def fails_with_no_message():
        raise ValueError()

    said = watched(fails_with_no_message, "failed-bare")
    assert [signal for signal, _ in said] == ["error"], said
    assert said[0][1][0] == f"{STAMP} error The run failed: ValueError", said

    class _Wordless(Exception):
        def __str__(self):
            raise RuntimeError("an exception whose words cannot be had")

    def fails_without_words():
        raise _Wordless()

    said = watched(fails_without_words, "failed-wordless")
    assert [signal for signal, _ in said] == ["error"], said
    assert said[0][1][0] == f"{STAMP} error The run failed: _Wordless", said

    def stops():
        raise streams.WorkerCancelled()

    assert watched(stops, "stopped") == [("cancelled", [f"{STAMP} window The run stopped: a stop was asked for."])]

    def stops_over_a_failure():
        try:
            raise ValueError("a failure in flight")
        except ValueError:
            raise streams.WorkerCancelled()

    said = watched(stops_over_a_failure, "in-flight")
    assert [signal for signal, _ in said] == ["log", "log", "cancelled"], said
    held = said[0][1]
    assert held[0].startswith(f"{STAMP} warning Cancelled. There may be a failure that was still in flight"), held
    assert held[1] == f"{STAMP} error Traceback (most recent call last):", held
    assert f"{INDENT}ValueError: a failure in flight" in held, held
    assert held[-1] == f"{STAMP} window The run stopped: a stop was asked for.", held

    assert watched(lambda: "done", "finished") == [("result", [f"{STAMP} window The run finished."])]

    token = streams.CancelToken()

    def asked_too_late():
        token.requested.set()                     # asked after the run's last line: nothing takes it
        return "done"

    late = [f"{STAMP} warning {CANCEL_TOO_LATE_TEXT}", f"{STAMP} window The run finished."]
    assert watched(asked_too_late, "too-late", cancel=token) == [("log", late), ("result", late)]

    assert streams._REDIRECT.acquire(blocking=False), "another run holds the streams"
    try:
        said = watched(lambda: "done", "declined")
    finally:
        streams._REDIRECT.release()
    declined = f"{STAMP} warning Another task already owns the console; this run's output is not captured."
    assert said == [("log", [declined]), ("result", [declined, f"{STAMP} window The run finished."])], said


def test_a_run_whose_output_cannot_be_captured_fails_and_says_why(tmp_path, monkeypatch):
    """When the capture of a run's output cannot start (the thread its pump needs cannot be made), the run is a
    failure said as one, in the window's words, never swallowed: ``error`` carries it and ``finished`` follows; the
    run's function never runs; its text file says how it ended; and the capture's lock is free for the next run."""
    qt_app()

    class _NoThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(streams, "_Pump", _NoThread)
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    said, ran = [], []
    _run(lambda: ran.append("ran"), text, **{signal: (lambda *args, signal=signal: said.append((signal, args)))
                                             for signal in ("error", "cancelled", "result", "finished")})
    text.close()
    assert [signal for signal, _ in said] == ["error", "finished"], said
    exc, trace, noted, kept = said[0][1]
    words = "The run could not start: PRISM could not capture its output (can't start new thread)."
    assert (type(exc), str(exc), noted, kept) == (RuntimeError, words, False, None), said[0]
    assert trace.rstrip().endswith("RuntimeError: can't start new thread"), trace
    assert ran == []
    assert _lines(text.path)[0] == f"{STAMP} error The run failed: RuntimeError: {words}", _lines(text.path)
    assert streams._REDIRECT.acquire(blocking=False), "the capture's lock was kept"
    streams._REDIRECT.release()


def test_a_run_that_ends_with_no_outcome_says_so_before_it_is_finished(tmp_path):
    """A run can end with none of its three outcomes: a library that ends the process its own way
    (``SystemExit``) passes every handler of the run. Its text file still says that the run ended without saying
    how, in the window's words, before ``finished`` is said, and nothing else is said."""
    qt_app()
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    said = []

    def exits():
        print("Simulating batch 1 of 4")
        raise SystemExit(3)

    try:
        _run(exits, text, **{signal: (lambda *args, signal=signal: said.append((signal, _lines(text.path))))
                             for signal in ("log", "error", "cancelled", "result", "finished")})
    except SystemExit:
        pass
    text.close()
    assert said == [("finished", [f"{STAMP} info Simulating batch 1 of 4",
                                  f"{STAMP} window The run ended without saying how."])], said


def test_a_line_written_to_the_runs_streams_after_the_run_is_not_in_its_file(tmp_path):
    """Once the run has ended its streams are no longer the run's, though something the run made may still write to
    them: a handler a library made during the run, a progress bar the collector closes later. Such a line is not
    the run's, and it does not reach the run's text file after its outcome."""
    qt_app()
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    kept = []

    def run():
        kept.append(sys.stderr)                     # as a handler a library makes during the run holds it
        print("Simulating batch 1 of 4")
        return "done"

    _run(run, text)
    kept[0].write("a library's late line, written after the run\n")
    written = _lines(text.path)
    text.close()
    assert written == [f"{STAMP} info Simulating batch 1 of 4", f"{STAMP} window The run finished."], written


def test_the_worker_hands_on_the_kept_training_named_by_a_stop_or_a_failure():
    """The training stage names the training a stop or a failure in its fit kept: ``kept_fit`` on the exception it
    raises. The worker hands that note on, as ``cancelled``'s object and ``error``'s fourth, read off the exception
    it caught; a stop or a failure that names none hands on None."""
    qt_app()
    note = SimpleNamespace(id="20261008T190211-1", passes=62, resumed=False, saved_here=True, processes=1)

    def raising(exc, kept):
        def run():
            if kept is not None:
                exc.kept_fit = kept
            raise exc
        return run

    got = []
    for kept in (note, None):
        _run(raising(streams.WorkerCancelled(), kept), None,
             cancelled=lambda *args: got.append(("cancelled", args)), error=lambda *args: got.append(("error", args)))
        _run(raising(RuntimeError("the graphics card ran out of memory"), kept), None,
             cancelled=lambda *args: got.append(("cancelled", args)), error=lambda *args: got.append(("error", args)))
    assert [(signal, len(args)) for signal, args in got] == [("cancelled", 1), ("error", 4)] * 2, got
    assert [args[-1] for _signal, args in got] == [note, note, None, None], got
    assert got[0][1][0] is note and got[1][1][3] is note


def test_a_librarys_warning_reaches_the_run_at_its_own_level():
    """A library's log record reaches a run through the window's root sink at its own level, as PRISM's own records
    do: a warning as a warning (the run's page shows it among its messages), an information record as information,
    an error as an error. Written as text on the error stream, a warning would take that stream's level, which is no
    warning."""
    from core import logging_root
    from core.gui.app import _library_record_sink

    qt_app()
    library = logging.getLogger("sbi.inference")
    level, heard = library.level, []
    library.setLevel(logging.INFO)
    logging_root.install(_library_record_sink)
    try:
        with streams.redirect_streams(WorkerSignals(), line_sink=lambda words, at: heard.append((words, at))):
            library.warning("Only 0.5% proposal samples are accepted.")
            library.info("Training neural network.")
            library.error("The density estimator diverged.")
            logging.warning("Only 0.4% proposal samples are accepted.")
    finally:
        logging_root.remove()
        library.setLevel(level)
    assert heard == [("library: sbi.inference: Only 0.5% proposal samples are accepted.", "warning"),
                     ("library: sbi.inference: Training neural network.", "info"),
                     ("library: sbi.inference: The density estimator diverged.", "error"),
                     ("library: test_run_text: Only 0.4% proposal samples are accepted.", "warning")], heard


def test_another_threads_record_that_cannot_be_written_writes_nothing_on_the_runs_streams(monkeypatch):
    """A record another thread logs during a run goes to the console the run stands in front of. When that console
    cannot take it, the failure is reported on the console's own error stream and never on the run's: the run's
    streams are the process's for the run's time, and a report written there would land in the run's text from a
    thread that is not the run's. With no console error stream the report is dropped. Nothing is raised."""
    import io

    qt_app()
    for err in (io.StringIO(), None):
        narrow = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        monkeypatch.setattr(sys, "stdout", narrow)
        monkeypatch.setattr(sys, "stderr", err)
        heard = []
        with streams.redirect_streams(WorkerSignals(), line_sink=lambda words, level: heard.append(words)):
            other = threading.Thread(
                target=lambda: logging.getLogger("core.gui.main_window").info("the χ probe table holds no row"))
            other.start()
            other.join()
        assert heard == [], heard
        if err is not None:
            assert "--- Logging error ---" in err.getvalue() and "UnicodeEncodeError" in err.getvalue(), err.getvalue()


# ── the history file ─────────────────────────────────────────────────────────────────────────────
FIELDS = ("id", "word", "name", "place", "records_root", "started", "ended", "last_heard", "outcome", "stop_reason",
          "stage", "acknowledged", "tries", "plan", "settings", "memory_limit", "simulation_set", "health", "fit",
          "check", "result", "failure", "text")


def _at(day, hour=12, minute=0, second=0):
    """A local moment of October 2026."""
    return datetime(2026, 10, day, hour, minute, second).astimezone()


def _ids(records):
    return [record["id"] for record in records]


def test_one_history_file_per_run_holds_what_its_run_did(tmp_path, monkeypatch):
    """A run's history file holds every field of the history, in its order, None where the run has nothing, its
    times local with their offset; it reads back as it was written. It is written whole: to a temporary file that is
    made to reach the disk and then renamed over the run's file, so a write that fails, even at the rename, leaves
    the last whole file and no temporary one, and a temporary file or a torn file is never read as a run. What the
    history does not keep, at the top or inside the set, the fit or a try, and a time with no zone, are refused
    before anything is written."""
    from core.gui.run import history

    started, ended = _at(8, 19, 2, 11), _at(14, 14, 10)
    record = {
        "id": "20261008T190211-training", "word": "Training", "name": "retrain_3", "place": ["inference", "posterior"],
        "records_root": str(tmp_path / "records"), "started": started, "ended": ended, "last_heard": None,
        "outcome": "stopped", "stop_reason": None, "stage": "Fit", "acknowledged": False,
        "tries": [{"started": started, "ended": ended, "outcome": "stopped", "failure": None, "batches_before": 0,
                   "batches_after": 10000, "epochs_before": 0, "epochs_after": 62}],
        "plan": "retrain", "settings": {"num_runs": 10000, "hidden_features": 128, "num_transforms": 5},
        "memory_limit": "The graphics card's memory limit in force is 6 GiB, set in Settings.",
        "simulation_set": {"folder": "eda5c2a1e468", "batches": 10000, "at_start": 0, "at_end": 10000,
                           "per_batch": 2048, "seconds_per_batch": 9.7},
        "health": {"failed_simulations": 3, "memory_peak_gib": 5.8},
        "fit": {"epochs": 62, "best_epoch": 58, "best_validation_loss": -3.39812, "seconds_per_epoch": 3780.0,
                "ended_by": "stopped", "network": [128, 5], "simulations": 20480000, "kept_id": "20261014T141000-1",
                "passes": 62, "resumed": False, "processes": 1, "wall_seconds": 234360.0, "save_error": None},
        "check": None, "result": [], "failure": None, "text": "20261008T190211-training.txt"}
    replaced, synced, refusals, real_replace = [], [], [], os.replace

    def replace(source, target):
        replaced.append((Path(source).name, Path(target).name, len(synced)))
        if refusals:
            raise refusals.pop()
        real_replace(source, target)

    monkeypatch.setattr(history.os, "replace", replace)
    monkeypatch.setattr(history.os, "fsync", synced.append)
    folder = tmp_path / "runs"
    path = history.write(folder, record)
    assert path == folder / "20261008T190211-training.json"
    assert replaced == [("20261008T190211-training.json.tmp", "20261008T190211-training.json", 1)], (
        f"renamed (source, target, requests of the disk before it): {replaced}")
    assert sorted(p.name for p in folder.iterdir()) == ["20261008T190211-training.json"]
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(on_disk) == history.FIELDS == FIELDS
    assert history.read(path) == on_disk == {**record, "started": started.isoformat(), "ended": ended.isoformat(),
                                             "tries": [{**record["tries"][0], "started": started.isoformat(),
                                                        "ended": ended.isoformat()}]}
    assert on_disk["started"].startswith("2026-10-08T19:02:11") and datetime.fromisoformat(
        on_disk["started"]).utcoffset() == started.utcoffset(), on_disk["started"]

    history.write(folder, {"id": "20261009T080000-prior", "word": "Prior", "started": _at(9, 8)})
    sparse = history.read(folder / "20261009T080000-prior.json")
    assert tuple(sparse) == FIELDS and (sparse["word"], sparse["outcome"], sparse["fit"]) == ("Prior", None, None)

    before = path.read_bytes()
    for refused in ({**record, "colour": "red"}, {**record, "fit": {**record["fit"], "pace": 1.0}},
                    {**record, "simulation_set": {**record["simulation_set"], "secs_per_batch": 9.7}},
                    {**record, "tries": [{**record["tries"][0], "epochs_kept": 62}]},
                    {**record, "started": datetime(2026, 10, 8, 19, 2, 11)}, {**record, "id": ""}):
        try:
            history.write(folder, refused)
        except ValueError:
            pass
        else:
            raise AssertionError(f"written: {refused}")
    try:
        history.write(folder, {**record, "text": {"a set, which the history cannot write"}})
    except TypeError:
        pass
    else:
        raise AssertionError("a record the history cannot write was written")
    refusals.append(OSError(28, "No space left on device"))
    try:
        history.write(folder, {**record, "outcome": "finished"})
    except OSError:
        pass
    else:
        raise AssertionError("a rename that failed was not raised")
    assert path.read_bytes() == before, "a failed write touched the run's file"
    assert sorted(p.name for p in folder.iterdir()) == ["20261008T190211-training.json", "20261009T080000-prior.json"]

    (folder / "20261010T090000-training.json.tmp").write_text('{"id": "20261010T0900', encoding="utf-8")
    (folder / "20261011T090000-training.json").write_text('{"id": "20261011T09', encoding="utf-8")
    assert history.read(folder / "20261011T090000-training.json") is None
    assert _ids(history.latest(folder, records_root=str(tmp_path / "records"), n=9)) == ["20261008T190211-training"]


def test_a_rename_another_program_holds_up_is_tried_again(tmp_path, monkeypatch):
    """On Windows a rename over a file that another program holds open (a scanner, an indexer, an editor) is
    refused for a moment. The history tries it again, ``REPLACE_TRIES`` times in all, ``REPLACE_PAUSE_S`` seconds
    apart, so the run's end is still recorded; a refusal that outlasts every try is raised, the run's file left as
    it was and no temporary file left behind."""
    from core.gui.run import history

    tries, paused, refusing, real_replace = [], [], [0], os.replace

    def replace(source, target):
        tries.append(Path(target).name)
        if refusing[0]:
            refusing[0] -= 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another "
                                      "process")
        real_replace(source, target)

    monkeypatch.setattr(history.os, "replace", replace)
    monkeypatch.setattr(history, "_pause", paused.append)
    folder = tmp_path / "runs"
    record = {"id": "20261008T190211-training", "word": "Training", "started": _at(8, 19, 2, 11)}
    refusing[0] = 2
    path = history.write(folder, {**record, "outcome": "finished"})
    assert (len(tries), paused) == (3, [history.REPLACE_PAUSE_S] * 2), (tries, paused)
    assert history.read(path)["outcome"] == "finished"
    before, refusing[0] = path.read_bytes(), history.REPLACE_TRIES
    tries.clear()
    paused.clear()
    try:
        history.write(folder, {**record, "outcome": "failed"})
    except PermissionError:
        pass
    else:
        raise AssertionError("a refusal that outlasted every try was not raised")
    assert (len(tries), len(paused)) == (history.REPLACE_TRIES, history.REPLACE_TRIES - 1), (tries, paused)
    assert path.read_bytes() == before, "a refused write touched the run's file"
    assert sorted(p.name for p in folder.iterdir()) == ["20261008T190211-training.json"]
    assert (history.REPLACE_TRIES, history.REPLACE_PAUSE_S) == (5, 0.2)


def test_the_runs_folder_sits_beside_the_settings_file(tmp_path):
    """The runs' files sit in the folder ``runs`` beside the settings file: here the test's own, which the suites'
    guard hands every test, so a test never writes beside the person's. Naming the folder makes nothing."""
    from core.gui import settings
    from core.gui.run import history

    assert Path(settings.settings().fileName()) == tmp_path / "prism.ini", "the suites' settings guard is not in force"
    assert history.runs_folder() == tmp_path / "runs"
    assert not (tmp_path / "runs").exists()


def test_a_run_id_is_its_local_start_and_its_word(tmp_path):
    """A run's id is its local start, to the second, and its word lower-cased with its spaces as hyphens. A second
    run of the same word in the same second takes ``-2``, then ``-3``, so no run's files are written over."""
    from core.gui.run import history

    started = _at(8, 19, 2, 11)
    assert history.run_id("Training", started, tmp_path) == "20261008T190211-training"
    assert history.run_id("Narrowing round", started, tmp_path) == "20261008T190211-narrowing-round"
    (tmp_path / "20261008T190211-training.json").write_text("{}", encoding="utf-8")
    assert history.run_id("Training", started, tmp_path) == "20261008T190211-training-2"
    (tmp_path / "20261008T190211-training-2.txt").write_text("", encoding="utf-8")
    assert history.run_id("Training", started, tmp_path) == "20261008T190211-training-3"
    assert history.run_id("Training", _at(8, 19, 2, 12), tmp_path) == "20261008T190212-training"


def test_the_latest_runs_are_those_of_the_records_folder_in_use(tmp_path):
    """Home's last runs are read for the records folder in use: a run of another records folder is not among them,
    however new. Newest first, as many as asked for; the same folder spelled another way is the same folder."""
    from core.gui.run import history

    here, there, folder = tmp_path / "records", tmp_path / "elsewhere", tmp_path / "runs"
    for run, day, root in (("20261001T120000-training", 1, here), ("20261002T120000-prior", 2, there),
                           ("20261003T120000-prior", 3, here), ("20261004T120000-training", 4, here),
                           ("20261005T120000-training", 5, there)):
        history.write(folder, {"id": run, "word": "Training", "started": _at(day), "outcome": "finished",
                               "records_root": str(root)})
    assert _ids(history.latest(folder, records_root=str(here), n=2)) == [
        "20261004T120000-training", "20261003T120000-prior"]
    assert _ids(history.latest(folder, records_root=str(here), n=9)) == [
        "20261004T120000-training", "20261003T120000-prior", "20261001T120000-training"]
    assert _ids(history.latest(folder, records_root=str(there), n=9)) == [
        "20261005T120000-training", "20261002T120000-prior"]
    assert _ids(history.latest(folder, records_root=str(there / ".." / "records"), n=1)) == [
        "20261004T120000-training"]
    assert history.latest(tmp_path / "nothing", records_root=str(here), n=3) == []


_LOADS = """
import sys
sys.path.insert(0, sys.argv[1])
import core.gui.run.history
print(sorted(name for name in ("torch", "core.SBI.training_checkpoint") if name in sys.modules))
"""


def test_a_failed_training_can_be_continued_from_its_sets_last_saved_batch(tmp_path, monkeypatch):
    """A training of the records folder in use that failed or was not seen to end can be continued while its
    simulation set is unfinished: from the set's last saved batch, read off the set's folder once the run has
    ended. Not a run that was stopped (its own page offers that), nor one still running (the engine rewrites the
    set's state as it goes), nor one whose set is complete or holds no saved batch, nor once a later training of
    the records folder in use has started, nor a run of another records folder. The set's state is read by a module
    that loads torch, imported only when it is asked for: the history itself loads no torch, as a fresh interpreter
    shows."""
    from core.SBI import training_checkpoint
    from core.artifacts.store import ArtifactStore
    from core.gui.run import history

    states = {"eda5c2a1e468": {"batches_done": 6150, "complete": False},
              "0a1b2c3d4e5f": {"batches_done": 10000, "complete": True},
              "5f4e3d2c1b0a": {"batches_done": 0, "complete": False}}
    read = []

    def peek(path):
        read.append(Path(path))
        return states.get(Path(path).name)

    monkeypatch.setattr(training_checkpoint, "peek", peek)
    store, folder = ArtifactStore(tmp_path / "records"), tmp_path / "runs"

    def training(run, day, outcome, set_folder="eda5c2a1e468", root=None):
        record = {"id": run, "word": "Training", "started": _at(day), "outcome": outcome,
                  "records_root": str(root or store.root), "simulation_set": {"folder": set_folder}}
        history.write(folder, record)
        return record

    failed = training("20261003T120000-training", 3, "failed")
    assert history.continuable(failed, store=store, folder=folder) == 6150
    assert read == [store.kind_dir("simulation") / "eda5c2a1e468"], read
    assert history.continuable({**failed, "outcome": "not seen to end"}, store=store, folder=folder) == 6150
    read.clear()
    for outcome in ("stopped", "finished", "running"):
        assert history.continuable({**failed, "outcome": outcome}, store=store, folder=folder) is None, outcome
    assert history.continuable({**failed, "records_root": str(tmp_path / "elsewhere")}, store=store,
                               folder=folder) is None
    assert read == [], f"the set was read for a run that cannot be continued: {read}"
    for set_folder in ("0a1b2c3d4e5f", "5f4e3d2c1b0a", "ffffffffffff"):
        assert history.continuable({**failed, "simulation_set": {"folder": set_folder}}, store=store,
                                   folder=folder) is None, set_folder
    training("20261004T120000-training", 4, "running", root=tmp_path / "elsewhere")
    history.write(folder, {"id": "20261004T130000-prior", "word": "Prior", "started": _at(4, 13),
                           "outcome": "finished", "records_root": str(store.root)})
    assert history.continuable(failed, store=store, folder=folder) == 6150, "a later run of another kind or folder"
    training("20261005T120000-training", 5, "finished", set_folder="0a1b2c3d4e5f")
    assert history.continuable(failed, store=store, folder=folder) is None, "a later training has started"

    fresh = subprocess.run([sys.executable, "-c", _LOADS, str(REPO)], cwd=REPO, capture_output=True, text=True,
                           timeout=120)
    assert (fresh.returncode, fresh.stdout.strip()) == (0, "[]"), fresh.stdout + fresh.stderr


def test_the_estimate_is_the_newest_pace_of_a_run_of_the_same_word_and_size(tmp_path):
    """Until a run has its own pace, its page borrows the newest pace of an ended run of the same word and size,
    and says whose it is. Per batch: the same simulations per batch. Per epoch: the same network fitted on the same
    number of simulations. A run still going, one of another size or word, and one that ended before any pace, are
    passed over; with none left, there is no estimate."""
    from core.gui.run import history

    folder = tmp_path / "runs"

    def run(run_id, day, *, word="Training", name="", outcome="finished", per_batch=2048, seconds=None,
            network=(128, 5), simulations=20480000, per_epoch=None):
        history.write(folder, {"id": run_id, "word": word, "name": name, "started": _at(day), "outcome": outcome,
                               "simulation_set": {"folder": "eda5c2a1e468", "per_batch": per_batch,
                                                  "seconds_per_batch": seconds},
                               "fit": {"network": list(network), "simulations": simulations,
                                       "seconds_per_epoch": per_epoch}})

    run("20261001T120000-training", 1, name="retrain_2", seconds=10.4, per_epoch=3600.0)
    run("20261002T120000-training", 2, name="retrain_aug", outcome="stopped", seconds=9.7, per_epoch=3780.0)
    run("20261003T120000-training", 3, name="retrain_3", outcome="running", seconds=8.8, per_epoch=3000.0)
    run("20261004T120000-training", 4, per_batch=1024, seconds=5.1, network=(64, 5), simulations=1024000,
        per_epoch=50.0)
    run("20261005T120000-narrowing-round", 5, word="Narrowing round", seconds=7.7, per_epoch=1800.0)
    run("20261006T120000-training", 6, name="no_pace", outcome="failed")
    by_batch, by_epoch = {"per_batch": 2048}, {"network": [128, 5], "simulations": 20480000}
    assert history.estimate(folder, word="Training", per="batch", size=by_batch) == (
        9.7, "the last training of this size, retrain_aug, took 9.7 s per batch")
    assert history.estimate(folder, word="Training", per="epoch", size=by_epoch) == (
        3780.0, "the last training of this size, retrain_aug, took about 63 minutes per epoch")
    assert history.estimate(folder, word="Training", per="epoch", size={"network": (64, 5), "simulations": 1024000}) \
        == (50.0, "the last training of this size took about 50 seconds per epoch")
    assert history.estimate(folder, word="Narrowing round", per="batch", size=by_batch) == (
        7.7, "the last narrowing round of this size took 7.7 s per batch")
    for word, per, size in (("Training", "batch", {"per_batch": 4096}), ("Prior", "batch", by_batch),
                            ("Training", "epoch", {"network": [256, 10], "simulations": 20480000}),
                            ("Training", "epoch", {"network": [128, 5], "simulations": 10240000}),
                            ("Training", "batch", {}), ("Training", "epoch", {"network": [128, 5]})):
        assert history.estimate(folder, word=word, per=per, size=size) is None, (word, per, size)
    assert history.estimate(tmp_path / "nothing", word="Training", per="batch", size=by_batch) is None


def test_the_runs_of_a_simulation_set_are_found_by_its_folder(tmp_path):
    """Records and a run's page find the runs that wrote to a simulation set by the set's folder, oldest first; a
    run of another set, or of none, is not among them."""
    from core.gui.run import history

    folder = tmp_path / "runs"
    for run, day, set_folder in (("20261003T120000-training", 3, "eda5c2a1e468"),
                                 ("20261001T120000-training", 1, "eda5c2a1e468"),
                                 ("20261002T120000-narrowing-round", 2, "0a1b2c3d4e5f"),
                                 ("20261004T120000-prior", 4, None)):
        history.write(folder, {"id": run, "word": "Training", "started": _at(day), "outcome": "finished",
                               "simulation_set": {"folder": set_folder} if set_folder else None})
    assert _ids(history.runs_for(folder, "eda5c2a1e468")) == ["20261001T120000-training", "20261003T120000-training"]
    assert _ids(history.runs_for(folder, "0a1b2c3d4e5f")) == ["20261002T120000-narrowing-round"]
    assert history.runs_for(folder, "ffffffffffff") == []


def test_a_run_still_marked_running_is_closed_as_not_seen_to_end(tmp_path):
    """A history file that still says "running" when PRISM opens is a run that ended unseen (one PRISM runs at a
    time): it is rewritten as not seen to end, heard last when its file was last written, and its id is handed
    back. A run that ended is left as it is, byte for byte, and a second closing finds nothing."""
    from core.gui.run import history

    folder = tmp_path / "runs"
    died = history.write(folder, {"id": "20261003T010000-training", "word": "Training", "started": _at(3, 1),
                                  "outcome": "running", "stage": "Simulation"})
    also = history.write(folder, {"id": "20261004T090000-prior", "word": "Prior", "started": _at(4, 9),
                                  "outcome": "running"})
    ended = history.write(folder, {"id": "20261005T090000-training", "word": "Training", "started": _at(5, 9),
                                   "outcome": "finished"})
    for path, heard in ((died, _at(3, 3, 12)), (also, _at(4, 10))):
        os.utime(path, (heard.timestamp(), heard.timestamp()))
    kept = ended.read_bytes(), ended.stat().st_mtime_ns
    assert history.close_unseen(folder) == ("20261003T010000-training", "20261004T090000-prior")
    closed = history.read(died)
    assert (closed["outcome"], closed["last_heard"], closed["stage"]) == (
        "not seen to end", _at(3, 3, 12).isoformat(), "Simulation"), closed
    assert history.read(also)["last_heard"] == _at(4, 10).isoformat()
    assert (ended.read_bytes(), ended.stat().st_mtime_ns) == kept
    assert history.close_unseen(folder) == ()
