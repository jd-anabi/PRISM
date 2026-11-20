"""A run in a test: the stand-in run, the window's clock moved by hand, the run's full text as the controller
says it, and the turning of the event loop until something is so.

The stand-in run is a plain function the real worker runs through a page's real ``dispatch``: every line it says
reaches the run as an engine's line does, through the ``core`` logger or the standard streams, and passes the
points where a requested stop is taken as an engine line does. Between its lines it waits on its gate, never by a
sleep of its own; a test says what it says next (``say``, ``row``, ``figure``, ``solver``) and how it ends
(``release``, ``fail``). Asked to stop while it waits, it says one more line, at which the stop is taken, as an
engine stops at its next line. Told nothing for a minute, it raises, so a run its test never released cannot hold
the process at its end.

The helpers the run suites share stand beside the stand-in: a moment of the test's day (``at``), the controller on a
test's own clock and folder (``controller_on``), a training as its page hands it to its run (``training_spec``), a
training brought to a stage by the engine's own lines (``say``, ``to_the_simulation``, ``to_the_fit``) and those lines
(``writing``, ``complete``, ``epoch_line``).

A stand-in tells the ``on_stage`` it is handed of its stages as the engine's staged calls (the card test's) do, each
in its turn among its lines (``stage``) or all at once as it is released (``stages``), and the engine's lines of a
check and of a narrowing round stand ready to be said: ``masked_line`` and ``masked_none`` (the training's total of
unusable driven recordings, in each form it is written), ``progress_frame``, ``CALIBRATION_LINES``, ``region_line``
and ``ROUND_LINES``.

A stop asked from a point inside a run is asked on the window's thread, as the window's own controls ask it
(``WindowsStop``), and a test can note every stop asked from any other thread (``stops_off_the_windows_thread``); a
run's worker can be made to end with none of the three outcomes (``runs_that_end_unsaid``).

At module level this file imports the standard library and the toolkit's core, nothing of torch and nothing of the
program: what it needs of the program it imports where it needs it.
"""
import logging
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal

#: Where the window places each panel's runs, by the panel's class: what a stand-in run on such a panel is told,
#: unless its test hands it a spec of its own. A bare run host has no place; a panel of any other class is refused by
#: ``start_run`` when no spec is handed.
PLACES = {"ConfigPanel": ("inference", "model"), "PriorPanel": ("inference", "prior"),
          "PosteriorPanel": ("inference", "training"), "ValidatePanel": ("inference", "checks"),
          "InferPanel": ("inference", "infer"), "TSNPEPanel": ("inference", "narrowing")}
#: The line a stand-in run asked to stop says while it waits, at which the stop is taken.
STOP_LINE = "The stand-in run reaches its next line."
#: How long a stand-in run waits, with nothing told it, before it gives up and raises: a run its test never
#: released ends, and holds no thread of the pool when the process ends.
WAIT_LIMIT_S = 60.0
#: How long a stand-in run that does not take a stop by itself holds one, once the end of its test has asked it to
#: stop, before it takes it anyway at its own line: a test that failed before it had the run say its line still sees
#: the run stop at its end, as the end of every test asks. Until the end of the test asks, a held stop waits for the
#: test's own line however long that takes.
HELD_STOP_S = 5.0
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
#: The window's local time zone in a test, seven hours behind UTC: no test reads the computer's own.
_ZONE = timezone(timedelta(hours=-7))
#: A training's lines as the engine writes them, which a stand-in run says to be brought to a stage of its run.
BUDGET = "[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows"
ROTATING = "Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)..."
ROTATED = "[fisher] eigenvalue spread (best/worst direction): 412"
NOT_SAVED = "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed."
#: What a stand-in's stage told "failed" raises when no error of the stage's own is handed with it.
STAGE_FAILED_TEXT = "stand-in stage failed"
#: The events the engine's staged calls report: a stage is skipped, begins, ends with its seconds, or fails with them.
STAGE_EVENTS = ("skipped", "started", "finished", "failed")


def _stage_item(stage, event, seconds=None, raises=None) -> tuple:
    """One event of a staged call as the stand-in keeps it, ``("stage", stage, event, seconds, raises)``. ValueError
    for one the engine never tells: a stage the card test does not have, an event that is none of ``STAGE_EVENTS``,
    seconds for a stage skipped or begun (the engine tells those with none), or an error to raise at an event other
    than "failed"."""
    from core.diagnostics.card import STAGES
    if stage not in STAGES:
        raise ValueError(f"a stage is one of the card test's {STAGES}, not {stage!r}")
    if event not in STAGE_EVENTS:
        raise ValueError(f"a stage's event is one of {STAGE_EVENTS}, not {event!r}")
    if seconds is not None and event in ("skipped", "started"):
        raise ValueError(f"the engine tells a stage {event} with no seconds, not {seconds!r}")
    if raises is not None and event != "failed":
        raise ValueError(f"only a stage told failed raises an error, not one told {event!r}")
    return ("stage", stage, event, seconds, raises)


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


def at(day, hour, minute, second=0) -> datetime:
    """A moment of October 2026 in the test's local time, with its zone (the 3rd is a Saturday, the 4th a Sunday, the
    8th a Thursday)."""
    return datetime(2026, 10, day, hour, minute, second, tzinfo=_ZONE)


class StandIn:
    """The stand-in run. ``started`` is set when the run begins; ``stopped_at`` is the line at which a stop was
    taken, or None, and ``stopped`` is set as the stop is taken. ``kept_fit`` is the note of a kept training its stop
    or its failure hands back, as the training stage hands one back (None for none). With ``takes_stops`` False it
    says no line of its own when a stop is asked for: the stop is taken at the next line its test has it say, or,
    once the end of the test has asked and none is said within ``HELD_STOP_S``, at its own line. A stop is asked for
    when the controller's status says so, and also when the run's own token is set with none asked through the
    controller, as the end of a test sets it for a training saving its posterior. It may be called again, as a try of
    a run or the repeat of its call is: ``calls`` holds the arguments of each call, in order, and each call ends at
    the end its test gives it after the call began.

    A call handed an ``on_stage`` as a keyword (the card test's) tells it its stages on the thread the run is on, as the
    engine's staged call does: each event queued with ``stage`` is told in its turn among the lines, so that a test can
    look at the live run after any event and say lines between two. A stage told "started" and not yet "finished" or
    "failed" is told "failed", with the seconds since it started, when the call ends any other way (a stop, ``fail``,
    the wait's limit), before that end passes on, as the engine tells a stage left by any way out of it. ``stages`` is
    the shorthand, ``(stage, event, seconds)`` each, told in order as the run is released, just before it returns; with
    it no stage is told while the run is live, no line comes between two events, a stop or a ``fail`` tells nothing,
    and a "failed" raises ``RuntimeError(STAGE_FAILED_TEXT)``, never the stage's own error: a test that needs any of
    these queues its events with ``stage``. A call handed no ``on_stage`` is told nothing and passes over its stage
    items, a "failed" one among them."""

    def __init__(self, *, word: str = "Training", lines=(), takes_stops: bool = True, stages=()):
        self.word = word
        self.takes_stops = takes_stops
        self.stages = tuple(_stage_item(stage, event, seconds)[1:4] for stage, event, seconds in stages)
        self.kept_fit = None
        self.calls: list[tuple[tuple, dict]] = []
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.stopped_at: str | None = None
        self._lock = threading.Lock()
        self._queue = deque(("say", text, level) for text, level in lines)
        self._wake = threading.Event()
        self._end = None
        self._said_stop = False
        self._open = None                     # the stage told "started" and not yet ended in this call, and when

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

    def stage(self, stage: str, event: str, seconds: "float | None" = None,
              raises: "BaseException | None" = None) -> None:
        """Queue one event of a staged call, told in its turn among the lines: ``stage`` one of the card test's stages
        (``core.diagnostics.card.STAGES``), ``event`` one of ``STAGE_EVENTS``. "skipped" and "started" carry no
        seconds; "finished" and "failed" carry ``seconds``, or, when it is None, the seconds since the stage was told
        "started" in this call. At "failed" the call then raises ``raises``, the stage's own error (a refusal, a lack
        of the card's memory), or ``RuntimeError(STAGE_FAILED_TEXT)`` when none is handed. ValueError, as it is
        queued, for an event the engine never tells (``_stage_item``)."""
        self._put(_stage_item(stage, event, seconds, raises))

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
        from tests import _built
        with self._lock:
            self.calls.append((args, dict(kwargs)))
        self._said_stop, self._open = False, None
        self.started.set()
        token = getattr(getattr(controller(), "_run", None), "token", None)       # this run's own
        log, saying, framed, ending_at = logging.getLogger("core"), None, False, None
        on_stage = kwargs.get("on_stage")
        heard = time.monotonic()
        try:
            while True:
                with self._lock:
                    items, end = list(self._queue), self._end
                    self._queue.clear()
                    if end is not None:
                        self._end = None                 # this call's end: a call after it waits for its own
                for item in items:
                    saying = item[1] if item[0] == "say" else None
                    if item[0] == "stage":
                        self._tell(item, on_stage)
                    else:
                        framed = self._do(item, log, fig_sink) or framed
                saying = None
                if items:
                    heard = time.monotonic()
                elif end is None and time.monotonic() - heard > WAIT_LIMIT_S:
                    raise RuntimeError(f"the stand-in run was told nothing for {WAIT_LIMIT_S:g} s and never released")
                if end is not None:
                    if end[0] == "fail":
                        if self.kept_fit is not None:
                            end[1].kept_fit = self.kept_fit
                        raise end[1]
                    self._pass_stages(on_stage)
                    return end[1]
                asked = controller().status.stop_requested or (token is not None and token.requested.is_set())
                # the held stop's fallback, armed only once the end of the test has asked
                ending_at = (ending_at or time.monotonic()) if asked and _built.ENDING.is_set() else None
                held = asked and (ending_at is None or time.monotonic() - ending_at < HELD_STOP_S)
                if asked and (self.takes_stops or not held) and not self._said_stop:
                    self._said_stop, saying = True, STOP_LINE
                    log.info(STOP_LINE)
                    saying = None
                elif not asked:
                    self._said_stop = False          # a stop withdrawn: the next one is taken at its line again
                self._wake.wait(0.01)
                self._wake.clear()
        except BaseException as leaving:
            if self._open is not None:           # a stage left by any other way out is told "failed" first
                stage, began = self._open
                self._open = None
                on_stage(stage, "failed", time.perf_counter() - began)
            if isinstance(leaving, WorkerCancelled):
                self.stopped_at = saying
                if self.kept_fit is not None:
                    leaving.kept_fit = self.kept_fit
                self.stopped.set()
            raise
        finally:
            if framed:
                sys.stderr.write("\r")           # the bar's row is let go, as a bar that does not stay goes

    def _pass_stages(self, on_stage) -> None:
        """The shorthand's stages, told in order on the thread this is called on (the run's), as queued ones are."""
        for stage, event, seconds in self.stages:
            self._tell(("stage", stage, event, seconds, None), on_stage)

    def _tell(self, item, on_stage) -> None:
        """Tell ``on_stage`` one stage event, on the run's thread, keeping which stage is open as the engine does: its
        "started" is told before the stage's work and its "finished" after it, so neither is a way out of the stage,
        and "failed" ends the stage, then raises the stage's error. Nothing at all when no ``on_stage`` was handed. A
        stage's seconds are read off the fine clock: the coarse one moves only every sixtieth of a second or so here."""
        if on_stage is None:
            return
        _, stage, event, seconds, raises = item
        if event == "started":
            began = time.perf_counter()
            on_stage(stage, event, None)
            self._open = (stage, began)
            return
        opened, self._open = self._open, None
        if seconds is None and event in ("finished", "failed"):
            seconds = time.perf_counter() - opened[1] if opened is not None and opened[0] == stage else 0.0
        on_stage(stage, event, seconds)
        if event == "failed":
            raise raises if raises is not None else RuntimeError(STAGE_FAILED_TEXT)

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
            sys.stderr.write("\r" + progress_frame(desc, done, total))
            return True
        elif kind == "figure":
            if fig_sink is not None:
                from matplotlib.figure import Figure
                fig_sink(item[1], Figure())
        elif kind == "solver":
            from core.progress import SOLVER
            SOLVER.add(item[1])
        return False


@contextmanager
def a_worker_that_cannot_be_made():
    """Inside the block the run's controller cannot make a run's worker: a start raises ``TypeError`` once every page is
    locked, its page handed the run and its status published, and before its worker is queued, as a start that
    cannot be finished does, and is taken back."""
    from core.gui.run.controller import RunController
    made = RunController._worker

    def refused(self, run, kwargs):
        raise TypeError("the run's worker cannot be made")

    RunController._worker = refused
    try:
        yield
    finally:
        RunController._worker = made


@contextmanager
def runs_that_end_unsaid():
    """Inside the block every run's worker the run's controller makes ends with none of the three outcomes, as a worker
    does when an exception of a kind no clause of the run takes, or the capture's own exit, passes every handler: it
    says one line of the run (``BUDGET``), writes as the run's outcome that it ended without saying how, and says
    ``finished`` alone. The run's own call never runs, and nothing leaves the pool's thread."""
    from core.gui.run import controller as made
    from core.gui.worker import UNSAID_TEXT, Worker

    class Unsaid(Worker):
        def run(self):
            self.signals.log_batch.emit([(BUDGET, "info")])
            if self.run_text is not None:
                self.run_text.outcome(UNSAID_TEXT, "window")
            self.signals.finished.emit()

    real = made.Worker
    made.Worker = Unsaid
    try:
        yield
    finally:
        made.Worker = real


class WindowsStop(QObject):
    """A stop asked of the run's controller from a point inside a run that its test chooses, and asked on the window's
    thread, as every control of the window asks one (the controller writes a stop's flags there, with no lock). Called
    on the run's thread, it hands the ask to the window's thread, which makes it at its next turn of the event loop
    (``turn_until`` turns it), and waits until it has been made: the run then goes on, its token set. Its keywords are
    the stop's own (``closing``, ``reason``). Made on the window's thread before its run starts, and let go when its
    test ends, as what the builders make is."""

    _asking = Signal(object)

    def __init__(self):
        super().__init__()
        from tests import _built
        self._asked = threading.Event()
        self._asking.connect(self._ask)
        _built.keep(self)

    def __call__(self, **kwargs) -> None:
        self._asked.clear()
        self._asking.emit(kwargs)
        if not self._asked.wait(WAIT_LIMIT_S):
            raise RuntimeError(f"the window's thread did not ask for the stop within {WAIT_LIMIT_S:g} s")

    # A method of this object's own: the queued call lands on the window's thread, where the object lives.
    def _ask(self, kwargs) -> None:
        from core.gui.run.controller import controller
        try:
            controller().request_stop(**kwargs)
        finally:
            self._asked.set()


def stops_off_the_windows_thread(monkeypatch) -> list:
    """Every stop asked of the run's controller from now on, to the test's end, by a thread that is not the window's:
    the thread's name, each, in the list handed back, and the stop asked as it was. A test reads the list at its end and
    finds none: the controller writes a stop's flags on the window's thread, with no lock."""
    from core.gui.run.controller import RunController
    off, real = [], RunController.request_stop

    def asked(held, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            off.append(threading.current_thread().name)
        return real(held, *args, **kwargs)

    monkeypatch.setattr(RunController, "request_stop", asked)
    return off


def start_run(host, stand_in, *, spec=None, **kwargs) -> None:
    """Start ``stand_in`` through ``host``'s own ``dispatch``, the host's ``run_spec`` standing in for this one
    start only: ``spec`` when it is handed, else a run of the stand-in's word at the place the window gives the host's
    class (``PLACES``), or at none for a bare run host (the class every ``build_run_host()`` makes, named ``P``). A
    host of any other class is refused with a ``KeyError`` naming it when no spec is handed: it would be started with
    no place, and every test that leaned on the default would stay green with none. ``kwargs`` are the dispatch's own
    (``provide_fig_sink``, ``on_result``). Fails when a run is live already."""
    from core.gui.run.model import RunSpec
    given = spec
    if given is None:
        kind = type(host).__name__
        if kind != "P" and kind not in PLACES:
            raise KeyError(f"the stand-in run has no place for a page of class {kind}: add it to PLACES")
        given = RunSpec(stand_in.word, None if kind == "P" else PLACES[kind], stop_button="Stop the run…")
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


def controller_on(tmp_path, clock):
    """The process's run controller, its clock and its folder of runs this test's own; the end of the test puts
    both back."""
    from core.gui.run.controller import controller
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    return held


def training_spec(word="Training", **changed):
    """A training, or with another ``word`` a narrowing round, as its page hands it to its run: the page's own
    ``training_spec`` is called, so the stages, the kinds of record, the tries again and the facts follow the page.
    It is called with a stand-in configuration holding what the page reads of one (the model, the kind of observation
    and whether the rotation is on) and with the knobs a call hands it: the name ``retrain_3``, a patience of 20
    epochs, a limit of 300, a network of 256 hidden features and 10 transforms, and no saving of the simulation set
    between batches, so that the spec does not depend on whether the engine's setting of it has been switched off yet.
    The place is the training step's, for both words. ``changed`` then replaces any field of the spec, whole, as
    ``dataclasses.replace`` does (``facts`` among them)."""
    from dataclasses import replace
    from types import SimpleNamespace
    from core.gui.panels.inference.base import training_spec as the_pages
    cfg = SimpleNamespace(model="NADROWSKI", observation_mode="chi", reparam_rotate=True)
    button = "Stop training…" if word == "Training" else "Stop the narrowing round…"
    kwargs = {"name": "retrain_3", "stop_after_epochs": 20, "max_num_epochs": 300, "hidden_features": 256,
              "num_transforms": 10, "checkpoint_every": 0}
    spec = the_pages(word, ("inference", "training"), button, cfg, kwargs, narrowed=word != "Training")
    return replace(spec, **changed)


def writing(folder) -> str:
    """The engine's line that names the simulation set's folder as the simulation begins."""
    return f"[checkpoint] writing to {folder} every 50 batches (~10.3 GiB total, 120.5 GiB free)"


def complete(folder) -> str:
    """The engine's line that says the simulation set in ``folder`` is complete."""
    return (f"[checkpoint] complete: 10000 batches in {folder}. Safe to delete once the posterior is saved; keeping "
            f"it lets you retrain the flow without re-simulating.")


def epoch_line(n, since=0, took=3780.0, training=-3.0, validation=-3.1) -> str:
    """An epoch's record, as the training call writes it: epoch ``n``, ``took`` seconds, ``since`` epochs since the
    best, and the two losses (written in the shortest form that keeps six digits, so -3.0 reads -3)."""
    return (f"[fit] epoch {n}: training loss {training:.6g}, validation loss {validation:.6g}, took {took:.1f} s, "
            f"epochs since the best {since}")


# ── the engine's lines of a check and of a narrowing round ──
#: The scopes a training's total of unusable driven recordings names, as the engine words them: every batch of the
#: simulation set was counted, or only this process's batches, with no set kept.
SCOPE_WHOLE = "every committed batch of the simulation cache"
SCOPE_PROCESS = "this process only (no simulation cache)"
#: The spread of the share across batches that every total made here states after its counts: lowest, highest and
#: median, in per cent. They are fixed figures, not worked out from the counts handed in.
_PER_BATCH = (31.2, 48.9, 40.0)


def scope_part(first: int) -> str:
    """The third scope of the total: a resumed simulation whose first ``first`` batches were committed with no
    per-batch counts, so the total counts only the batches this process generated. It holds a semicolon of its own."""
    return (f"only the batches this process generated; the cache's first {first} batches were committed without "
            f"per-batch counts")


def masked_line(unusable: int, total: int, batches: int, scope: str) -> str:
    """The training's total of unusable driven recordings as the engine words it: the counts with thousands separators,
    the share to one decimal, the number of batches, the ``scope`` (``SCOPE_WHOLE``, ``SCOPE_PROCESS`` or
    ``scope_part(n)``), then the spread across batches in the fixed figures of this module. ValueError for a total or a
    count of batches that is not above zero, or more unusable than the total."""
    if total <= 0 or batches <= 0 or not 0 <= unusable <= total:
        raise ValueError(f"a total of {unusable} unusable of {total} over {batches} batches is no total the engine says")
    low, high, middle = _PER_BATCH
    return (f"[chi] masked probes: {unusable:,} of {total:,} ({100.0 * unusable / total:.1f}%) over {batches} batches, "
            f"{scope}; per batch {low:.1f}-{high:.1f}%, median {middle:.1f}%")


def masked_none(scope: str) -> str:
    """The same record when no probe was counted."""
    return f"[chi] masked probes: none counted, {scope}"


def progress_frame(desc: str, done: int, total: int) -> str:
    """One frame of a progress bar as the engine's bar writes it on the error stream, without the carriage return that
    brings the cursor back to the start of its row: the description, the share done, the bar, ``done`` of ``total`` and
    the time."""
    pct = int(100 * done / total) if total else 0
    return f"{desc}: {pct:3d}%|{'#' * (pct // 10):<10}| {done}/{total} [00:10<00:20,  9.70s/it]"


#: The four lines a calibration's stages begin at, as the engine writes them. The first is a frame of the bar that counts
#: the calibration's simulations, which a run shows with ``stand_in.row``; the other three are records, said with
#: ``say``. The run total ends with the size beyond which the engine counts a trajectory as failed, 1e15.
CALIBRATION_LINES = (
    progress_frame("Generating training data", 3, 10),
    "[patho] run total: 0 pathological of 2,000 simulated trajectories (0.0000%) -- 0 non-finite, 0 exactly constant, "
    "0 over 1e+15",
    "SBC uniformity checks:",
    "TARP: ATC=0.0123  KS p=0.512",
)
#: The simulation set a narrowing round writes to, as the round's lines name it.
ROUND_FOLDER = r"C:\Users\J\PRISM\Artifacts\simulations\3f9a2c1d7b04"
#: The region a narrowing round's lines name, as the region's own class writes itself.
ROUND_REGION = "TruncationRegion(level=0.9, d0:[-1.2,1.1], d2:[-0.5,0.8], basis=unrotated)"


def region_line(kept: "str | None" = None) -> str:
    """A narrowing round's first line, its region: drawn from the observation ``obs_1`` for a new round, or, for a kept
    round resumed, read from the kept training whose id is ``kept``."""
    if kept is None:
        return f"[tsnpe] region from observation obs_1: {ROUND_REGION}"
    return f"[tsnpe] region of the kept training {kept}: {ROUND_REGION}"


#: A new narrowing round's lines, in the order the engine says them: its region, its simulation set, its total of
#: unusable driven recordings and its first epoch. A kept round resumed says ``region_line(<its id>)`` first instead.
ROUND_LINES = (
    region_line(),
    writing(ROUND_FOLDER),
    masked_line(1201, 3000, 3, SCOPE_WHOLE),
    epoch_line(1),
)


def say(app, stand_in, text, *lines, level="info", again=True) -> None:
    """Have the stand-in run say ``lines``, and turn the event loop until the run's full text holds the last. With
    ``again`` the text must hold it once more than it did before the call, so that a run that follows an earlier one of
    the same test, which said the same line, is waited for; without it one earlier is enough."""
    said = text.texts().count(lines[-1]) if again else 0
    for line in lines:
        stand_in.say(line, level)
    turn_until(app, lambda: text.texts().count(lines[-1]) > said)


def to_the_simulation(app, stand_in, text, held, folder, done, *, again=True) -> None:
    """A training brought to its simulation in ``folder``, ``done`` of its 10,000 batches simulated."""
    from core.gui.run.model import SIMULATION_ROW
    say(app, stand_in, text, BUDGET, ROTATING, ROTATED, writing(folder), again=again)
    stand_in.row(SIMULATION_ROW, done, 10_000)
    turn_until(app, lambda: held.model.batches_done() == done)


def to_the_fit(app, stand_in, text, clock, folder, epochs, *, saved=True, again=True) -> None:
    """A training brought to its fit on the set in ``folder``, one epoch's time later, its records ``epochs``
    ((epoch, since the best), ...) said; its fit not saved as it goes when not ``saved``."""
    say(app, stand_in, text, BUDGET, ROTATING, ROTATED, writing(folder), *(() if saved else (NOT_SAVED,)),
        complete(folder), again=again)
    clock.advance(3780)
    if epochs:
        say(app, stand_in, text, *(epoch_line(epoch, since) for epoch, since in epochs), again=again)


def note(kept_id, passes=62, resumed=False, saved_here=True, processes=1):
    """The note a training's stop or failure hands back of the training it kept."""
    from core.SBI.fit_state import KeptNote
    return KeptNote(kept_id, passes, resumed, saved_here, processes)


def ended(app, held) -> None:
    """Turn the event loop until the run has ended."""
    turn_until(app, lambda: not held.is_live(), 10)


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
