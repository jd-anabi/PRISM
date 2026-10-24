"""The run's text and its history file.

Every finished line a run writes reaches a dated text file as it is written, whichever of the three channels it
came by (a print, a record, a Python warning); the file's writer can never raise into a run, speak, or take a
requested stop inside a save; the worker writes how a run ended before it says anything to the window; a library's
record reaches a run at its own level; and each run keeps one history file beside the settings file.

No window is built here and no test sleeps: a run is a worker run on the test's own thread, every moment is read
off a clock the test holds, and a child process is waited for by reading what it prints.

Run: pytest tests/test_run_text.py
"""
import logging
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
    stamp's width, under its text."""
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    text.head(word="Training", name="retrain_3", run="20261008T190211-training",
              memory_limit="The graphics card's memory limit in force is 6 GiB, set in Settings.")
    text.separator(2)
    text.line('Traceback (most recent call last):\n  File "pipeline.py", line 12, in run\nRuntimeError: the batch '
              'failed', "error")
    text.close()
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
        f"{INDENT}RuntimeError: the batch failed"], _lines(text.path)
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
    """STANDING. The run's line sink is called on the thread that wrote the line, under the stream pump's lock, and
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
    ``close`` always does."""
    synced = []
    monkeypatch.setattr(textfile.os, "fsync", synced.append)
    clock = _Clock()
    text = RunTextFile(tmp_path / "run.txt", clock=clock)
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
    assert textfile.FSYNC_EVERY_S == 30.0
    assert len(_lines(text.path)) == 41


def test_a_pump_whose_window_is_gone_keeps_no_more_lines_but_the_file_still_gets_them(tmp_path):
    """A run whose window has gone (its signals deleted while it goes on) stops the pump at the first signal it
    cannot emit, and from then on the pump keeps no line for the window: they would pile up unread for the rest of
    the run. The run's text file still gets every line."""
    import shiboken6

    qt_app()
    text = RunTextFile(tmp_path / "run.txt", clock=_Clock())
    signals = WorkerSignals()
    pump = streams._Pump(signals, line_sink=text.line)
    shiboken6.delete(signals)
    pump.sink("log", ("Simulating batch 1 of 4", "info"))
    pump.stop()                                              # its last emit meets the deleted signals
    pump.sink("log", ("Simulating batch 2 of 4", "info"))
    pump.sink("log", ("[checkpoint] writing batches 0 to 2 of 4", "warning"))
    assert pump._logs == [], f"a pump with no window kept {pump._logs}"
    assert _lines(text.path) == [f"{STAMP} info Simulating batch 1 of 4", f"{STAMP} info Simulating batch 2 of 4",
                                 f"{STAMP} warning [checkpoint] writing batches 0 to 2 of 4"], _lines(text.path)


def test_the_worker_writes_its_outcome_before_it_says_anything(tmp_path):
    """How a run ended is in its text file before the worker says anything to the window: a slot on each signal
    reads the file as the signal arrives. A failure is its kind and its message, at error, its traceback indented
    under it; a stop and a finish are one line each, the window's own. The lines the worker says itself, outside the
    pump, are written too, each before it is said: the failure a stop may have replaced and its traceback, a stop
    that came too late to stop anything, and the redirect declined because another run holds the streams."""
    qt_app()

    def watched(fn, name, cancel=None):
        text = RunTextFile(tmp_path / f"{name}.txt", clock=_Clock())
        said = []

        def slot(signal):
            return lambda *args: said.append((signal, _lines(text.path)))

        _run(fn, text, cancel=cancel, **{signal: slot(signal) for signal in ("log", "error", "cancelled", "result")})
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
    assert [signal for signal, _ in said] == ["error", "finished"], said
    exc, trace, noted, kept = said[0][1]
    words = "The run could not start: PRISM could not capture its output (can't start new thread)."
    assert (type(exc), str(exc), noted, kept) == (RuntimeError, words, False, None), said[0]
    assert trace.rstrip().endswith("RuntimeError: can't start new thread"), trace
    assert ran == []
    assert _lines(text.path)[0] == f"{STAMP} error The run failed: RuntimeError: {words}", _lines(text.path)
    assert streams._REDIRECT.acquire(blocking=False), "the capture's lock was kept"
    streams._REDIRECT.release()


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
