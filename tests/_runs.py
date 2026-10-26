"""A run in a test: the stand-in run, the window's clock moved by hand, the run's full text as the controller
says it, and the turning of the event loop until something is so.

The stand-in run is a plain function the real worker runs through a page's real ``dispatch``: every line it says
reaches the run as an engine's line does, through the ``core`` logger or the standard streams, and passes the
points where a requested stop is taken as an engine line does. Between its lines it waits on its gate, never by a
sleep of its own; a test says what it says next (``say``, ``row``, ``figure``, ``solver``) and how it ends
(``release``, ``fail``). Asked to stop while it waits, it says one more line, at which the stop is taken, as an
engine stops at its next line. Told nothing for a minute, it raises, so a run its test never released cannot hold
the process at its end.

At module level this file imports the standard library and the toolkit's core, nothing of torch and nothing of the
program: what it needs of the program it imports where it needs it.
"""
import logging
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta

from PySide6.QtCore import QEventLoop, QObject, QTimer

#: Where the window places each panel's runs, by the panel's class: what a stand-in run on such a panel is told,
#: unless its test hands it a spec of its own. A bare run host has no place.
PLACES = {"ConfigPanel": ("inference", "model"), "PriorPanel": ("inference", "prior"),
          "PosteriorPanel": ("inference", "training"), "ValidatePanel": ("inference", "checks"),
          "InferPanel": ("inference", "infer"), "TSNPEPanel": ("inference", "narrowing"),
          "FdtPanel": ("temperature", "one_cell"), "CrossValPanel": ("temperature", "sweep"),
          "ReductionPanel": ("reduction", None), "SimulatePanel": ("live", None)}
#: The line a stand-in run asked to stop says while it waits, at which the stop is taken.
STOP_LINE = "The stand-in run reaches its next line."
#: How long a stand-in run waits, with nothing told it, before it gives up and raises: a run its test never
#: released ends, and holds no thread of the pool when the process ends.
WAIT_LIMIT_S = 60.0
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class Clock:
    """The window's clock in a test: local time with its zone, moved by hand."""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("the window's clock gives local time with its zone")
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class StandIn:
    """The stand-in run. ``started`` is set when the run begins; ``stopped_at`` is the line at which a stop was
    taken, or None."""

    def __init__(self, *, word: str = "Training", lines=()):
        self.word = word
        self.started = threading.Event()
        self.stopped_at: str | None = None
        self._lock = threading.Lock()
        self._queue = deque(("say", text, level) for text, level in lines)
        self._wake = threading.Event()
        self._end = None
        self._said_stop = False

    # ── what a test tells it, from the test's thread ──
    def _put(self, item) -> None:
        with self._lock:
            self._queue.append(item)
        self._wake.set()

    def say(self, text: str, level: str = "info") -> None:
        """Queue a line and let one step of the run pass: at ``"info"``, ``"warning"`` or ``"error"`` a record of
        the ``core`` logger; ``"print"`` a print on the output stream; ``"err"`` a print on the error stream."""
        self._put(("say", text, level))

    def row(self, desc: str, done: int, total: int) -> None:
        """Queue one frame of a progress bar, ``done`` of ``total``, written over itself on the error stream as the
        engine's bar writes it."""
        self._put(("row", desc, done, total))

    def figure(self, title: str) -> None:
        """Queue a figure handed to the run's figure sink, when the run was given one: a bare figure of the figure
        library, never through its state machine."""
        self._put(("figure", title))

    def solver(self, steps: int) -> None:
        """Queue steps of the solver, counted on the run's own thread as the solver counts them."""
        self._put(("solver", steps))

    def release(self, result=None) -> None:
        """End the run: it returns ``result``."""
        with self._lock:
            self._end = ("release", result)
        self._wake.set()

    def fail(self, exc: BaseException) -> None:
        """End the run: it raises ``exc``."""
        with self._lock:
            self._end = ("fail", exc)
        self._wake.set()

    # ── the run itself, on the worker's thread ──
    def __call__(self, *args, fig_sink=None, **kwargs):
        from core.gui.run.controller import controller
        from core.gui.streams import WorkerCancelled
        self.started.set()
        log, saying, framed = logging.getLogger("core"), None, False
        heard = time.monotonic()
        try:
            while True:
                with self._lock:
                    items, end = list(self._queue), self._end
                    self._queue.clear()
                for item in items:
                    saying = item[1] if item[0] == "say" else None
                    framed = self._do(item, log, fig_sink) or framed
                saying = None
                if items:
                    heard = time.monotonic()
                elif end is None and time.monotonic() - heard > WAIT_LIMIT_S:
                    raise RuntimeError(f"the stand-in run was told nothing for {WAIT_LIMIT_S:g} s and never released")
                if end is not None:
                    if end[0] == "fail":
                        raise end[1]
                    return end[1]
                if controller().status.stop_requested and not self._said_stop:
                    self._said_stop, saying = True, STOP_LINE
                    log.info(STOP_LINE)
                    saying = None
                self._wake.wait(0.01)
                self._wake.clear()
        except WorkerCancelled:
            self.stopped_at = saying
            raise
        finally:
            if framed:
                sys.stderr.write("\r")           # the bar's row is let go, as a bar that does not stay goes

    @staticmethod
    def _do(item, log, fig_sink) -> bool:
        """Do one queued thing; True when it wrote a frame of a bar."""
        kind = item[0]
        if kind == "say":
            _, text, level = item
            if level == "print":
                print(text)
            elif level == "err":
                print(text, file=sys.stderr)
            else:
                log.log(_LEVELS[level], text)
        elif kind == "row":
            _, desc, done, total = item
            pct = int(100 * done / total) if total else 0
            sys.stderr.write(f"\r{desc}: {pct:3d}%|{'#' * (pct // 10):<10}| {done}/{total} [00:10<00:20,  9.70s/it]")
            return True
        elif kind == "figure":
            if fig_sink is not None:
                from matplotlib.figure import Figure
                fig_sink(item[1], Figure())
        elif kind == "solver":
            from core.progress import SOLVER
            SOLVER.add(item[1])
        return False


def start_run(host, stand_in, *, spec=None, **kwargs) -> None:
    """Start ``stand_in`` through ``host``'s own ``dispatch``, the host's ``run_spec`` standing in for this one
    start only: ``spec`` when it is handed, else a run of the stand-in's word at the place of the host's kind
    (none for a bare run host). ``kwargs`` are the dispatch's own (``watch_dir``, ``provide_fig_sink``,
    ``on_result``). Fails when a run is live already."""
    from core.gui.run.model import RunSpec
    given = spec if spec is not None else RunSpec(stand_in.word, PLACES.get(type(host).__name__),
                                                  stop_button="Stop the run…")
    host.run_spec = lambda fn, *args, **kw: given
    try:
        started = host.dispatch(stand_in, **kwargs)
    finally:
        del host.run_spec
    if not started:
        raise AssertionError("the stand-in run was not started: a run was live already")


def turn_until(app, done, limit: float = 5.0) -> None:
    """Turn the event loop until ``done()`` is true; AssertionError after ``limit`` seconds. The loop waits for its
    next event, woken at least every few milliseconds, and sleeps nowhere."""
    began = time.monotonic()
    waker = QTimer()
    waker.start(2)
    try:
        while not done():
            if time.monotonic() - began > limit:
                raise AssertionError(f"not so after {limit:g} s")
            app.processEvents(QEventLoop.AllEvents | QEventLoop.WaitForMoreEvents)
    finally:
        waker.stop()


@contextmanager
def running(host=None, stand_in=None, **kwargs):
    """A stand-in run live for the block: started on ``host`` (a bare run host when none is handed), released when
    the block ends, and turned until its end has been handled."""
    from core.gui.run.controller import controller
    from tests._fixtures import build_run_host, qt_app
    app = qt_app()
    host = build_run_host() if host is None else host
    stand_in = StandIn() if stand_in is None else stand_in
    start_run(host, stand_in, **kwargs)
    try:
        yield stand_in
    finally:
        stand_in.release()
        turn_until(app, lambda: not controller().is_live(), 10.0)


class RunText(QObject):
    """What the run's controller says while it is heard: each line of the run's full text as ``(level, text)``,
    in order (``lines``), each status published (``statuses``) and how each run ended (``ended``). Made before the
    run it hears, and let go when its test ends, as what the builders make is."""

    def __init__(self, controller):
        super().__init__()
        from tests import _built
        self.lines: list[tuple[str, str]] = []
        self.statuses: list = []
        self.ended: list = []
        controller.lines.connect(self._take)
        controller.status_changed.connect(self._status)
        controller.run_ended.connect(self._end)
        _built.keep(self)

    # Methods of this object's own, each: a deleted object leaves the process's controller no connection.
    def _take(self, batch) -> None:
        self.lines.extend((level, text) for text, level in batch)

    def _status(self, status) -> None:
        self.statuses.append(status)

    def _end(self, status) -> None:
        self.ended.append(status)

    def texts(self, level: str | None = None) -> list[str]:
        """The lines' texts, of ``level`` alone when one is handed."""
        return [text for at, text in self.lines if level is None or at == level]
