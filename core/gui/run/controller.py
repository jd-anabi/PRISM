"""The run's controller: the one run of the process, from the start a page asks for to the end it leaves behind.

There is one per process (``controller()``), because a run's output is captured for the whole process and only one
run goes at a time; every window hears it, each through methods of its own, so that a window deleted leaves no
connection behind. A page starts a run with ``dispatch``: the controller writes the run's history file and its text
file (where the run keeps them), makes the run's model, locks the controls of every panel, says the run is starting
and queues the worker. All the worker says reaches the controller on the window's thread, where the model is fed,
the run's state worked out again, the history written (at each change of state or stage, at each save of the
simulation set the run passes, at each epoch and at the end) and every line handed on to the run's full text
(``lines``). At the end the panels are unlocked and the payload let go, the run's status is published, then
``run_ended``; only then is the run's page drawn ended.

A run's failure is a state of its page, never a box: the page says what failed and when, what was saved and what
was lost, and quotes the error's message, its kind and traceback shown when asked for. A training's simulation set
is read for the batches it holds once the run's worker has ended, never while it runs. A run that only loads a
record has no page of its own: its failure is told as a failure at a click is.

After the graphics card ran out of memory, a training or a narrowing round is tried again (``core/gui/run/retry.py``
says when): the run waits, Retrying, for a pause read on the run's clock, then its own call goes again in a new
worker with a new token, in the same text and history files, the set continued or the kept training resumed;
"Stop trying" (``stop_trying``) ends the tries, and the run ends failed, and so does a stop asked for in the turn
the pause runs out. The page of a training that failed or was stopped offers "Continue from batch N"
(``continue_run``) or "Resume the training" (``resume_training``): its own call again, as the page's next run, but
only into the session the call was made with. Its call is kept for that, until the page starts its next run (a load,
which is handed no page, leaves it kept) or goes, or the offer is pressed once the page's session holds another
model or prior; wherever the call goes, the page's offer goes with it. Every other run lets go of its call at its
end.

A refusal the stage makes before anything is spent is no run: the solver has taken no step since the start, no
stage after the run's first has begun, and the store holds no record of a kind the run writes made since the start.
Its two files are removed, the controller is idle again with nothing to acknowledge, and the page shows the refusal
as it shows one made at the click.

A run that only loads a record, a live simulation and a video export keep no file and end straight into idle, with
nothing to acknowledge.

A stop asked for (``request_stop``) is taken at the run's next point where a stop is taken and can be withdrawn
until then (``withdraw_stop``); asked by the window's close, it carries the close, and the window closes once the
run has ended. A training saving its posterior is not stopped, the save being a thing no stop interrupts: the
window's close then waits for the save with no stop, and can be withdrawn until the run ends (``withdraw_close``). A
stop asked with a reason of its own, Windows ending the session, is said in the run's text and its history from the
moment it is asked, the history saying the run was stopped and why; a training saving its posterior is not stopped by
it either, and its history says so only should it not finish. A
training a stop kept stays kept until its page's Discard deletes it (``discard``), once the run has ended.

While a run is live the computer is kept awake, from its start to its end, and Windows is given the reason it should
not end the session, registered on the newest window built (``use_window``) from the run's first line to its end and
set again at each change of the run's stage (``core/gui/this_computer.py``); both are asked from the window's thread,
from the controller's slots and its own timer, and both stand through a stop and a wait to be tried again.

The run's page is its host's: the controller hands it the run at the start and draws it again from the run's model
at every change the model hears and at each look; the run's full text reaches it through ``lines``. A run that only
loads a record is handed no page: it never takes the page's place, nor the page of a run that shows there. Nothing
here draws: a figure a run hands over goes to its page's stack of pictures as the picture it already is.
"""
import json
import logging
import math
import os
import statistics
import threading
import weakref
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from core.refusals import Refusal

from .. import this_computer
from ..streams import CancelToken
from ..worker import Worker
from . import ends, history, questions, retry
from .model import FIT_PACE_RECORDS, IDLE, RunModel, RunSpec, RunStatus, moved
from .textfile import FSYNC_EVERY_S, SEPARATOR_TEXT, RunTextFile

log = logging.getLogger(__name__)

#: How often, in milliseconds, the controller looks at a live run: the model's silence, the solver's count, the
#: text file's request to the disk.
TICK_MS = 1000

#: Said among a page's own lines when it asks for a run while one is live.
LIVE_TEXT = "A run is live. Only one runs at a time."
#: The run's line on the memory limit in force for each batch of simulations, filled with the limit and where it
#: was set; and the line when there is none.
LIMIT_TEXT = "Memory limit: {} GiB for each batch of simulations ({})."
NO_LIMIT_TEXT = "No memory limit is set on this computer."
#: Said in the run's full text when its history file cannot be written; filled with the error's kind and message.
HISTORY_BROKEN_TEXT = ("The run's history file could not be written ({}); the run goes on, and the file is tried "
                       "again at the run's next change.")
#: Said in the run's full text when, once the run has ended, its history file cannot be written again to say that
#: its end was seen or that the training it kept was discarded; filled with the error's kind and message.
HISTORY_LEFT_TEXT = ("The run's history file could not be written again after the run's end ({}); it says what it "
                     "said before.")
#: The window's log, when a run's page cannot be drawn in its past view: the history file's path and why.
PAST_UNDRAWN_TEXT = "The page of a run could not be drawn from its history file %s (%s)."


def _local_now() -> datetime:
    """The computer's clock, as local time with its zone."""
    return datetime.now().astimezone()


def _memory_limit() -> tuple[float | None, str]:
    """The memory limit in force for each batch of simulations, in GiB (None when there is none), and the run's
    line that says it and where it was set: by the variable of the environment, or in the window's step 1."""
    from core.SBI import pipeline
    gib = pipeline.vram_ceiling_gib()
    if gib <= 0:
        return None, NO_LIMIT_TEXT
    raw = os.environ.get(pipeline.VRAM_CEILING_ENV)
    try:
        by_variable = raw is not None and bool(raw.strip()) and float(raw) >= 0
    except ValueError:
        by_variable = False
    source = f"set by {pipeline.VRAM_CEILING_ENV}" if by_variable else "set in step 1, Model and options"
    return gib, LIMIT_TEXT.format(f"{gib:g}", source)


def _out_of_memory():
    """The engine's rule of a failure that is the graphics card running out of memory, loaded when a run has failed
    and never at import: the rule's module loads the graphics library. The rule handed back answers False for an
    error the engine's rule cannot read (one whose message cannot be had), so that no error holds up a run's end."""
    from core.SBI.pipeline import _is_oom

    def is_oom(exc) -> bool:
        try:
            return bool(_is_oom(exc))
        except Exception:                        # noqa: BLE001 -- an error the rule cannot read is no lack of memory
            return False
    return is_oom


def _made(created) -> datetime | None:
    """A record's moment as its manifest gives it, or None when it cannot be read as one with its zone."""
    try:
        moment = datetime.fromisoformat(created)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


def _asked(kwargs: dict) -> str:
    """The name the person typed for the record a run writes, as its call was handed it: the call's ``name``, or
    the name of the record its page made at the click (a measurement's ``writer``). "" when the call was handed
    none: a check, and a training, a narrowing round, a prior's build and an inference started from the window, are
    named after they are saved, never by their call; a run's own word for itself ("Calibration") is no name."""
    for given in (kwargs.get("name"), getattr(kwargs.get("writer"), "name", None)):
        if isinstance(given, str) and given:
            return given
    return ""


def _expected(store, facts: dict) -> tuple[int, str] | None:
    """The epochs the newest finished training of this kind ran, with the words that say where the number comes
    from; None when there is none. A training of this kind is a posterior of the same model, the same kind of
    observation and the same narrowing (the three facts the posterior's record holds: its configuration's model,
    its body's mode and whether it is amortized); one whose record holds no count of epochs is passed over. The
    store is listed once."""
    from core.artifacts import StoreError
    from core.artifacts.store import MANIFEST
    from .. import record_words
    try:
        listed = store.list("posterior")
    except (OSError, StoreError):
        return None
    for summary in listed:                                   # complete first, then the newest first
        if (not summary.complete or summary.mode != facts.get("observation")
                or summary.amortized != (not facts.get("narrowed"))):
            continue
        try:
            manifest = json.loads((Path(summary.path) / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or (manifest.get("config") or {}).get("model") != facts.get("model"):
            continue
        epochs = ((manifest.get("body") or {}).get("training") or {}).get("epochs_trained")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            continue
        made = _made(summary.created)
        label = record_words.record_label("posterior", summary.name, None if made is None else made.astimezone())
        return epochs, f"about {epochs} epochs: the last training of this kind, {label}, ran {epochs}"
    return None


class _Text:
    """What the worker writes as the run's text: the run's text file, when the run keeps one. ``line`` is the run's
    line sink, called under the stream pump's lock: it only writes the file (the window hears the same line through
    the worker's own signals). ``outcome``, how the run ended, is written to the file and kept as ``ended``: the
    controller says it on its ``lines`` when it hears the run's end, after every line the worker said before it,
    so the run's full text holds it where the file does."""

    def __init__(self, file: RunTextFile | None):
        self._file = file
        self.ended: tuple[str, str] | None = None

    def line(self, text: str, level: str) -> str | None:
        return None if self._file is None else self._file.line(text, level)

    def outcome(self, text: str, level: str, detail: str = "") -> None:
        if self._file is not None:
            self._file.outcome(text, level, detail)
        self.ended = (f"{text}\n{detail}" if detail else text, level)


class _Run:
    """What the controller holds of one run: the live one, and the last one until the next starts."""

    def __init__(self, spec: RunSpec, host, store, folder: Path):
        self.spec, self.host, self.store, self.folder = spec, host, store, folder
        self.id: str | None = None                 # its files' stem; None for a run that keeps none
        self.text: RunTextFile | None = None
        self.writes = spec.keeps_files             # whether its history file is still written
        self.history_said = False                  # whether a failed write of its history file was said
        self.sink: _Text | None = None             # what its worker writes its text to
        self.worker = None
        self.token: CancelToken | None = None
        self.watchers: list = []
        self.solver_at_start = 0
        self.real_start: datetime | None = None    # the computer's own time at the start, for the store's records
        self.outcome: str | None = None            # "finished", "stopped" or "failed"
        self.failure = None                        # (exception, traceback, cancel noted) of a failed run
        self.kept_fit = None
        self.continue_from: int | None = None      # the batches its unfinished simulation set holds, read at its end
        self.kept_now: tuple[int, int] | None = None    # (batches, epochs) its set and kept training held at its end
        self.model = None                          # its model, which its page draws
        self.keywords: dict = {}                   # its start's own keywords, for each try and a repeat of its call
        self.tried = 0                             # the tries made since it last kept more
        self.last_kept: tuple[int, int] | None = None   # (batches, epochs) kept at its last failure
        self.attempts: list[dict] = []             # each try that failed, as its history keeps it
        self.attempt_started: datetime | None = None     # when the try running began; None: the run's first
        self.attempt_before: tuple[int, int] | None = None   # (batches, epochs) kept when it began
        self.attempt_open = True                   # a try is running, or ended with the run: not yet in ``attempts``
        self.try_at: datetime | None = None        # when its next try begins, while it waits for it
        self.reopen = False                        # only reopening PRISM can help the failure it ended with
        self.failure_kind: str | None = None       # "out of memory", "device" or "other"
        self.repeats: str | None = None            # "continue" or "resume": the offer of a page this run repeats
        self.stop_requested = False
        self.stop_reason = ""                      # why the stop was asked, when not by the person: Windows' end
        self.reason_at: datetime | None = None     # when that reason was said in its text and its history
        self.closing = False                       # the window closes once the run has ended
        self.ended: datetime | None = None
        self.result: tuple | None = None           # (kind, id, name) of the record a finished run wrote
        self.asked = ""                            # the name its call was handed for that record ("" for none)
        self.record: dict | None = None            # the history its end wrote, which every later write starts from
        self.waiting: RunStatus | None = None      # the end that waited to be seen as it started, until it is set aside
        self.last: RunStatus | None = None         # its status at its end, which its page is drawn with after it
        self.set_named = False
        self.view = None                           # its host's run page, which it draws on
        self.at_start: int | None = None
        self.saves = 0
        self.epochs = 0
        self.synced: datetime | None = None


class RunController(QObject):
    """The one run of the process. ``status_changed`` carries the run's status at every change of it (and at each
    save of the simulation set the run passes); ``run_ended`` carries how the run ended, after the last status;
    ``lines`` carries each batch of ``(text, level)`` the run's full text receives, the head of its text file, its
    worker's own lines and how it ended among them."""

    status_changed = Signal(object)
    run_ended = Signal(object)
    lines = Signal(object)
    # A stop asked for, heard on the window's thread, from whichever thread asked; one asked for too late; and one
    # asked of a training saving its posterior.
    _stopping = Signal()
    _late = Signal()
    _unstoppable = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stopping.connect(self._stop_heard)
        self._late.connect(self._too_late)
        self._unstoppable.connect(self._cannot_stop)
        self.lines.connect(self._to_page)
        self._status: RunStatus = IDLE
        self._model: RunModel | None = None
        self._run: _Run | None = None
        self._clock = None                         # None: the computer's own clock
        self._folder: Path | None = None           # None: the folder of runs beside the settings file
        self._timer: QTimer | None = None          # made at the first run: nothing ticks before it
        self._spent_worker = None                  # the last run's worker, kept until the next run ends
        # The ended runs whose page offers to continue or resume them, by id, each with its own call kept: until its
        # page starts its next run, or goes.
        self._offers: dict[str, _Run] = {}
        self._window = None                        # the newest window built, held weakly: the reason goes on it
        self._awake = False                        # Windows is asked to keep the computer awake
        self._reason: tuple | None = None          # the live run's stage and the reason registered for it
        self._training_kept: int | None = None     # the batch the latest training of this process kept

    # ── what the shell and the pages read ──
    @property
    def status(self) -> RunStatus:
        """The run's status now."""
        return self._status

    @property
    def model(self) -> RunModel | None:
        """The model of the live run, or of the last one; None before the first."""
        return self._model

    def host(self):
        """The page the live run was started from, or the last run's; None before the first, and for a run no page
        started."""
        return None if self._run is None else self._run.host

    def is_live(self) -> bool:
        """Whether a run is live: starting, running, stopping or waiting to try again."""
        return self._status.live

    def now(self) -> datetime:
        """The moment now, on the clock the runs are read from."""
        return (self._clock or _local_now)()

    def folder(self) -> Path:
        """The folder of runs in use: the one a test handed in, else the one beside the settings file in use."""
        return self._folder if self._folder is not None else history.runs_folder()

    def training_kept(self) -> int | None:
        """The batch the latest training of this process kept in its simulation set when it was stopped or failed,
        which it can be continued from; None when it finished, kept no batch, or no training has ended."""
        return self._training_kept

    def use_clock(self, clock) -> None:
        """The clock every moment of the next runs is read from, local time with its zone; None for the
        computer's own."""
        self._clock = clock

    def use_folder(self, folder) -> None:
        """The folder of runs the next runs keep their files in, and their paces and earlier runs are read from;
        None for the one beside the settings file in use when each starts."""
        self._folder = None if folder is None else Path(folder)

    def use_window(self, window) -> None:
        """The window the reason against ending the session is registered on: the newest window built, which calls
        this as it is built. Held weakly: a window deleted is held by nothing here."""
        self._window = None if window is None else weakref.ref(window)

    # ── a run's start ──
    def dispatch(self, fn, *args, spec: RunSpec, host=None, on_result=None, provide_fig_sink: bool = False,
                 provide_stream: bool = False, on_chunk=None, watch_dir=None, **kwargs) -> bool:
        """Start ``fn(*args, **kwargs)`` as the process's run, described by ``spec``, for the page ``host``. False,
        and nothing started, while a run is live. Else, in this order: the memory limit in force is read once; the
        spec is given the call and the limit; its model; where the run keeps files, its id and its text file; the
        host's run page handed the run (none for a run that only loads a record); where the run keeps files, the
        text file's head, said on ``lines``, and its history file; the controls of every panel locked; the run's
        page shown in its host and drawn; the status published; the worker queued. A start that raises from the
        moment its page is handed the run to the moment its worker is queued is taken back -- every panel free, the
        host's results given back, the status idle again, the run's two files removed -- and the error is raised as
        it came: a page that cannot be drawn fails the start, and leaves nothing locked. ``on_result`` hears the
        run's result while the run is still live;
        ``provide_fig_sink`` hands the call a sink for its figures, which go to the host's stack of pictures;
        ``provide_stream`` hands it ``emit_chunk`` and ``should_stop``, and ``on_chunk`` hears each chunk;
        ``watch_dir`` is one folder, or several, whose new pictures go to the host's stack. None of these is handed
        on to ``fn``. An end that waits to be seen as the run starts is noted: the run sets it aside when it ends with
        an end of its own, and gives it back when it ends into idle or is taken back (``_set_aside``)."""
        if self.is_live():
            return False
        waiting = self._status if self._status.unacknowledged_end and self._status.run is not None else None
        from core.artifacts import default_store
        from core.progress import SOLVER
        from ..panels.base_panel import BasePanel
        from ..plot_watcher import NewPngWatcher
        clock = self._clock or _local_now
        gib, limit_text = _memory_limit()
        spec = spec.with_call(fn, args, kwargs)
        spec = replace(spec, facts={**spec.facts, "memory_limit": gib})
        store = default_store()
        folder = self._folder if self._folder is not None else history.runs_folder()
        run = _Run(spec, host, store, folder)
        run.asked, run.waiting = _asked(kwargs), waiting
        run.keywords = {"on_result": on_result, "provide_fig_sink": provide_fig_sink,
                        "provide_stream": provide_stream, "on_chunk": on_chunk}
        run.solver_at_start, run.real_start = SOLVER.steps, datetime.now(timezone.utc)
        # The page's next run: what its last run offered goes, with that run's call; so does the offer of a page that
        # has gone. A run that only loads a record is handed no page, and leaves the page's offer as it is.
        self._offers = {key: offered for key, offered in self._offers.items()
                        if _alive(offered.host) and (offered.host is not host or spec.loads_only)}
        expected = _expected(store, spec.facts) if "model" in spec.facts else None
        word = spec.word
        model = RunModel(spec, clock=clock, expected=expected,
                         estimate=lambda per, size: history.estimate(folder, word=word, per=per, size=size))
        self._model, self._run, run.model = model, run, model
        if spec.keeps_files:
            run.id = history.run_id(spec.word, model.started, folder)
            run.text = RunTextFile(folder / f"{run.id}.txt", clock=clock)
        try:
            # The host's run page is handed the run before anything is said on ``lines``: its full text holds the
            # head. A run that only loads a record is handed none: the page of the last run stays as it was.
            run.view = getattr(host, "run_view", None) if _alive(host) and not spec.loads_only else None
            if run.view is not None:
                run.view.begin(spec, text=None if run.text is None else run.text.path,
                               folder=folder if spec.keeps_files else None, clock=clock)
            if spec.keeps_files:
                said = [(f"{spec.word} {spec.name}, run {run.id}." if spec.name else f"{spec.word}, run {run.id}.",
                         "window"), (limit_text, "window")]
                broke = run.text.head(word=spec.word, name=spec.name, run=run.id, memory_limit=limit_text)
                if broke is not None:
                    said.append((broke, "warning"))
                    model.feed([(broke, "warning")])
                self.lines.emit(said)
                # After the head, so that a history that cannot be written from the first moment is said in the file.
                self._write(run)
            for panel in list(BasePanel._instances):
                panel.set_controls_enabled(False)
            if host is not None and hasattr(host, "_run_began"):
                host._run_began(spec)
            self._status = self._live_status()
            self._draw()
            if self._timer is None:
                self._timer = QTimer(self)
                self._timer.setInterval(TICK_MS)
                self._timer.timeout.connect(self.tick)
            self._timer.start()
            self._stay_awake(True)
            self.status_changed.emit(self._status)

            worker, token = self._worker(run, kwargs)
            stack = getattr(host, "figure_stack", None)
            # One watcher for each folder: a watcher reads one folder and not the folders under it, and a sweep
            # study writes two records. The controller holds them, so a page deleted during the run takes none
            # with it.
            for one in ([] if watch_dir is None else
                        [watch_dir] if isinstance(watch_dir, (str, Path)) else list(watch_dir)):
                watcher = NewPngWatcher(one, self)
                run.watchers.append(watcher)
                if stack is not None:
                    watcher.png_ready.connect(stack.add_png)
                watcher.start()
        except BaseException:
            self._take_back(run)
            raise
        run.worker, run.token = worker, token
        QThreadPool.globalInstance().start(worker)
        return True

    def _worker(self, run: _Run, kwargs: dict):
        """A worker for the run's own call, its keywords ``kwargs``, with a new token: the run's text file its line
        sink; its figures, its stream and its result handed where the run's start said; every signal it says heard
        here. Returns the worker and its token."""
        from ..panels.base_panel import _png_fig_sink
        keywords = run.keywords
        token, run.sink = CancelToken(), _Text(run.text)
        worker = Worker(run.spec.fn, *run.spec.args, cancel=token, run_text=run.sink, **kwargs)
        # Kept until the run has ended and its signals have been heard, and never deleted by the pool: a worker
        # the pool deleted as its run returned would take its still-queued signals with it.
        worker.setAutoDelete(False)
        signals = worker.signals
        if keywords.get("provide_fig_sink"):
            worker.kwargs["fig_sink"] = _png_fig_sink(signals.figure)
        if keywords.get("provide_stream"):
            # A long-lived streaming run hands each chunk to ``emit_chunk`` and polls ``should_stop`` to unwind.
            worker.kwargs["emit_chunk"] = signals.chunk.emit
            worker.kwargs["should_stop"] = token.requested.is_set
            if keywords.get("on_chunk") is not None:
                signals.chunk.connect(keywords["on_chunk"])
        stack = getattr(run.host, "figure_stack", None) if _alive(run.host) else None
        if stack is not None:
            signals.figure.connect(stack.add_figure)
        signals.log.connect(self._on_line)
        signals.log_batch.connect(self._on_lines)
        signals.rows.connect(self._on_rows)
        if keywords.get("on_result") is not None:
            signals.result.connect(keywords["on_result"])
        signals.result.connect(self._on_result)
        signals.error.connect(self._on_error)
        signals.cancelled.connect(self._on_cancelled)
        signals.finished.connect(self._on_finished)
        return worker, token

    def _take_back(self, run: _Run) -> None:
        """A start that raised before its worker was queued, taken back: its watchers let go, the timer stopped
        (once there is one), the run's two files removed, every panel free, the host's results given back (a load
        never took them), and the status published idle again, or, when an end waited to be seen as it started, that
        end's again: a start taken back sets nothing aside."""
        from ..panels.base_panel import BasePanel
        for watcher in run.watchers:
            watcher.deleteLater()
        run.watchers = []
        if self._timer is not None:
            self._timer.stop()
        if run.text is not None:
            self._remove_files(run)
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(True)
        self._stay_awake(False)
        if not run.spec.loads_only and _alive(run.host) and hasattr(run.host, "_run_over"):
            run.host._run_over(IDLE)
        self._status = run.waiting or IDLE
        self.status_changed.emit(self._status)

    # ── a run's stop and its end ──
    def request_stop(self, *, closing: bool = False, reason: str = "") -> None:
        """Ask the live run to stop at its next point where a stop is taken: the run is Stopping from now on, and its
        history says so. With ``closing`` the window closes once the run has ended; a stop asked for already gains
        that. With ``reason`` (Windows ending the session, for one) the stop is the same, and from now on the run's
        text says why it was asked and its history says the run was stopped and why: should the process be ended
        before the run is, both say so. A training saving its posterior is not stopped: with ``closing`` the window
        closes once the posterior is saved and the run has ended; with ``reason`` its text says the save goes on, and
        its history keeps the reason only should the run not finish; a stop asked for with neither comes too late and
        says so on the run's page. A stop asked of a run that has ended does nothing and says so on its page; nothing
        at all before the process's first run. Safe from any thread: the run's token is set at once, and the run's
        state, its history and its page follow on the window's thread (at once when asked there). Asked while the run
        waits to be tried again, it ends the tries, as "Stop trying" does."""
        run = self._run
        if reason and run is not None and self.is_live() and not run.stop_reason:
            run.stop_reason = reason
        if run is not None and run.try_at is not None and self.is_live():
            run.closing = run.closing or closing
            run.stop_requested = True
            self._stopping.emit()
            return
        token = None if run is None else run.token
        if token is None or not self.is_live():
            if run is not None and run.ended is not None:
                self._late.emit()
            return
        saving = self._model is not None and self._model.stage_kind() == "saving"
        if closing:
            run.closing = True
        elif saving and not run.stop_requested:
            if reason:
                self._stopping.emit()            # the save goes on: the reason is said and written all the same
            else:
                self._unstoppable.emit()
            return
        if not run.stop_requested and not saving:
            run.stop_requested = True
            token.requested.set()
        self._stopping.emit()

    def _stop_heard(self) -> None:
        """A stop or a close asked for, on the window's thread: a stop's reason of its own, the first time it is heard,
        said in the run's text and its full text and written into its history at once; the model told of a stop; and
        the run's state worked out again. For a run that waits to be tried again, its tries end at the event loop's
        next turn, after whoever asked has seen the stop asked for (the window's close waits for the end from
        there)."""
        run = self._run
        if run is None or not (run.stop_requested or run.closing or run.stop_reason) or not self.is_live():
            return
        if run.stop_reason and run.reason_at is None:
            run.reason_at = (self._clock or _local_now)()
            said = (questions.STOP_REASON_TEXT if run.stop_requested
                    else questions.SAVE_GOES_ON_TEXT).format(run.stop_reason)
            if run.text is not None:
                run.text.line(said, "window")
            self.lines.emit([(said, "window")])
            self._write(run)
        if run.try_at is not None:
            QTimer.singleShot(0, self, self._tries_end)
        elif run.stop_requested:
            self._model.stopping(True)
        self._refresh()
        self._draw()

    def stop_trying(self) -> None:
        """"Stop trying": the tries of a run that waits to be tried again end, and the run ends failed, as its last
        try failed. Nothing for any other run."""
        run = self._run
        if run is not None and run.try_at is not None and self.is_live():
            self.request_stop()

    def _tries_end(self) -> None:
        """The tries asked to end, on the window's thread: the run ends failed, said last in its text."""
        run = self._run
        if run is None or run.try_at is None or not self.is_live():
            return
        run.try_at, run.outcome = None, "failed"
        if run.text is not None:
            run.text.line(retry.STOPPED_TRYING_TEXT, "window")
        self.lines.emit([(retry.STOPPED_TRYING_TEXT, "window")])
        self._end(run, False)

    def _too_late(self) -> None:
        """A stop asked of a run that has ended, on the window's thread: its page says so."""
        view = None if self._run is None else self._run.view
        if _alive(view):
            view.say(questions.FINISHED_TEXT)

    def _cannot_stop(self) -> None:
        """A stop asked of a training saving its posterior, on the window's thread: its page says it came too late."""
        view = None if self._run is None else self._run.view
        if _alive(view):
            view.say(questions.SAVING_TEXT)

    def withdraw_stop(self) -> bool:
        """Withdraw the stop asked of the live run, and the close that waits for it, on the window's thread: True
        when the stop had not been taken, and the run goes on; False when the run had already begun to stop, and for
        a run whose stop cannot be withdrawn (one that asks nothing before it stops). The run's page says which."""
        run = self._run
        if run is None or run.token is None or not self.is_live() or not run.stop_requested or not run.spec.stop_asks:
            return False
        run.closing = False
        withdrawn = run.token.withdraw()
        if withdrawn:
            run.stop_requested = False
            self._model.stopping(False)
        self._refresh()
        self._draw()
        if _alive(run.view):
            run.view.say(questions.WITHDRAWN_TEXT.format(run.spec.word.lower()) if withdrawn
                         else questions.BEGUN_TO_STOP_TEXT)
        return withdrawn

    def withdraw_close(self) -> bool:
        """Withdraw the close that waits for the live run with no stop asked of it (a training saving its posterior),
        on the window's thread: True when there was one, and the window then stays open at the run's end, which the
        run's page says; False when there was none."""
        run = self._run
        if run is None or not self.is_live() or not run.closing or run.stop_requested:
            return False
        run.closing = False
        self._refresh()
        self._draw()
        if _alive(run.view):
            run.view.say(questions.KEPT_OPEN_TEXT)
        return True

    def discard(self, run: str | None) -> tuple[bool, str, str] | None:
        """Delete the training the stop of the run ``run`` kept, once that run has ended (the training's lock is let
        go by then): the one the controller holds, by the note its stop handed back, from the records it was kept
        in, or an earlier one, by the id its history kept, from the records that history names. ``(gone, words,
        ended)``: whether the training is gone, what the run's page says of it -- deleted, or no longer held by those
        records (both gone); not found, its run's history unreadable; or the store's own sentence of why it was not
        deleted (both kept) -- and, once it is gone, how the run ended and what it kept, as its page says it from
        then on ("" when the page's words stay as they are, ``_gone``). None while the run is live."""
        from core.artifacts import ArtifactStore, StoreError, default_store
        held = self._run
        if held is not None and held.id == run:
            if self.is_live():
                return None
            store, kept = held.store, getattr(held.kept_fit, "id", None)
        else:
            folder = self._folder if self._folder is not None else history.runs_folder()
            record = (None if run is None else history.read(folder / f"{run}.json")) or {}
            root = record.get("records_root")
            store = ArtifactStore(root) if root else default_store()
            kept = (record.get("fit") or {}).get("kept_id")
        if not kept:
            return False, questions.UNFOUND_TEXT, ""
        try:
            store.get("fit", kept)
        except StoreError:
            return True, questions.GONE_TEXT, self._gone(run)
        try:
            store.delete("fit", kept)
        except Refusal as refused:
            return False, refused.message, ""
        return True, questions.DISCARDED_TEXT, self._gone(run)

    def _gone(self, run: str | None) -> str:
        """The training the run ``run`` kept is gone: its page offers to resume it no more, and the run's call goes
        with the offer when its page offered nothing else; its history says it was discarded, nothing else changed
        (the history its end wrote, ``_write_ended``, for the run the controller holds; else its file, ``_rewrite``),
        and its end's words, read from that history, say so too: in the status while its end waits to be seen; in the
        end a live run noted as it started, which that run gives back as the status should it end into idle; and on
        its page, which is drawn here for the run the controller holds, and draws the words handed back for any
        other. They are dated against the moment the run ended for the run the controller holds, as its end dated
        them, and against the moment now for any other, as a run read back from its history is. A failed run's
        sentence stays, beside its error: "" is handed back for it, as when the history cannot be read again or put
        into words."""
        offered = self._offers.get(run)
        if offered is not None and offered.continue_from is None:
            del self._offers[run]
            self._let_go(offered)
        if run is None:
            return ""
        held, now = self._run, self.now()
        if held is not None and held.id == run:
            record = self._write_ended(held, _discarded)
            if held.last is None or record is None:
                return ""
            held.last = _said_again(held.last, record, held.ended or now)
            if self._status.run == run and self._status.unacknowledged_end:
                self._status = held.last
                self.status_changed.emit(self._status)
            self._draw()
            return "" if held.last.outcome == "failed" else held.last.sentence
        record = self._rewrite(self.folder(), run, _discarded)
        if record is None:
            return ""
        if self._status.run == run and self._status.unacknowledged_end:
            self._status = _said_again(self._status, record, now)
            self.status_changed.emit(self._status)
        if held is not None and held.waiting is not None and held.waiting.run == run:
            held.waiting = _said_again(held.waiting, record, now)
        try:
            sentence = ends.words(record, now)[0]
        except Exception:                        # noqa: BLE001 -- the page's words stay as its end said them
            return ""
        return "" if record.get("outcome") == "failed" else sentence

    def _let_go(self, run: _Run) -> None:
        """The run's own call let go, the configuration, the prior and the tensors its arguments hold with it: from
        the run, and from the model its page draws."""
        run.spec = replace(run.spec, fn=None, args=(), kwargs={})
        if run.model is not None:
            run.model.spec = run.spec
        run.keywords = {}

    def _withdraw_offer(self, run: str | None) -> None:
        """The offer of the page of the run ``run`` goes, in the controller and on the page, and the run's call with
        it."""
        offered = self._offers.pop(run, None)
        if offered is None:
            return
        self._let_go(offered)
        if _alive(offered.view):
            offered.view.offer(None, False)

    def _page_gone(self, run: str, *_signalled) -> None:
        """The page of the run ``run`` has gone (its panel deleted): its offer goes, and the run's call with it."""
        offered = self._offers.pop(run, None)
        if offered is not None:
            self._let_go(offered)

    def forget_offers(self) -> None:
        """Every call kept for a page's offer let go, and each offer withdrawn from its page, as though each such page
        had started its next run: no page offers to continue or resume a run any more, and the steps no longer say
        the batch the latest training kept."""
        for run in list(self._offers):
            self._withdraw_offer(run)
        self._training_kept = None

    def continue_run(self, run: str) -> bool:
        """"Continue from batch N" on the page of the training ``run``, which failed or was stopped with batches in
        its unfinished simulation set: its own call again, as its page's next run, the policy that must continue
        (``resume="require"``) the one new argument; the batch it continues from is the set's own to say. False, and
        nothing started, while a run is live, for a run no page offers to continue, and once the page's session holds
        another model or prior (``_repeat``)."""
        offered = self._offers.get(run)
        if offered is None or offered.continue_from is None:
            return False
        return self._repeat(offered, retry.continued_call(offered.spec), "continue")

    def resume_training(self, run: str) -> bool:
        """"Resume the training" on the page of the training ``run``, which was stopped and kept its training, or
        failed during its fit or while its posterior was saved: its own call again, as its page's next run, the kept
        training named (``resume_fit``) the one new argument. False, and nothing started, while a run is live, for a
        run no page offers to resume, and once the page's session holds another model or prior (``_repeat``)."""
        offered = self._offers.get(run)
        if offered is None or offered.kept_fit is None:
            return False
        return self._repeat(offered, retry.resumed_call(offered.spec, offered.kept_fit.id), "resume")

    def _repeat(self, offered: _Run, kwargs: dict, how: str) -> bool:
        """The call of the ended run ``offered`` started again from its page, with ``kwargs``, the keywords of its
        start as they were: a new run, whose page says, should the stage refuse it before anything is spent, why.
        Only into the session the call was made with: once the page's session holds another configuration or another
        prior (``retry.made_in``), nothing starts, the page says why, and its offer goes, with the call."""
        if self.is_live() or not _alive(offered.host):
            return False
        spec, keywords = offered.spec, offered.keywords
        if not retry.made_in(spec, getattr(offered.host, "session", None)):
            self._withdraw_offer(offered.id)
            if _alive(offered.view):
                offered.view.say(retry.CHANGED_TEXT)
            return False
        started = self.dispatch(spec.fn, *spec.args, spec=replace(spec, fn=None, args=(), kwargs={}),
                                host=offered.host, **keywords, **kwargs)
        if started:
            self._run.repeats = how
        return started

    def acknowledge(self, run: str) -> None:
        """Someone has seen how the run ``run`` ended: its history file says so, and when its end is the one the
        status says (the run the controller holds, or one of an earlier launch whose end the status was filled
        with), the controller is idle again. A run still live is not seen: its end is to come. An end seen while a
        later run is live is no longer that run's to set aside or give back."""
        self._write_seen(run)
        live = self._run
        if live is not None and live.waiting is not None and live.waiting.run == run:
            live.waiting = None
        if self._status.unacknowledged_end and self._status.run == run:
            self._status = IDLE
            self.status_changed.emit(IDLE)

    def _set_aside(self, run: _Run) -> None:
        """The run ``run`` has ended with an end of its own to say while the end that waited to be seen as it started
        still waits: the window says this run's end from now on, and never says two, so that end is written seen in
        its history, as though it had been looked at, and is not said again at the next launch. Only a run that keeps
        a history file and leaves an end of its own sets an end aside: one that ends into idle (a refusal before
        anything was spent, a load, a run that keeps no file) or is taken back gives that end back instead, as the
        status (``_end``, ``_take_back``)."""
        waiting, run.waiting = run.waiting, None
        if waiting is not None:
            self._write_seen(waiting.run)

    def _write_seen(self, run: str) -> None:
        """The run ``run`` written seen into its history, once it has ended: the run the controller holds, from the
        history its end wrote (``_write_ended``), or else an earlier run, from its history file in the folder of runs
        (``_rewrite``). Nothing for a run still live, which is seen only once it has ended: it has written no end
        yet."""
        held = self._run
        if held is not None and held.id == run:
            self._write_ended(held, {"acknowledged": True})
            return
        self._rewrite(self.folder(), run, {"acknowledged": True})

    def _write_ended(self, run: _Run, change) -> dict | None:
        """The history of the ended run ``run``, the run the controller holds, written again with ``change`` made
        (the names a dict sets, or what a function makes of the record) to the history its end wrote, and kept as its
        history: once a run has ended, its history is never worked out from the run again, so that a write after its
        end (its end seen, the training it kept discarded) keeps all its end wrote, its failure among it, and an end
        whose own write failed is written whole at last. The history now held is handed back; None for a run whose
        end wrote none. A write that changes nothing is not made; one that fails is said in the run's full text,
        once, as a history that cannot be written is said."""
        if run.record is None:
            return None
        changed = change(run.record) if callable(change) else {**run.record, **change}
        if changed == run.record:
            return run.record
        run.record = changed
        if run.writes:
            try:
                history.write(run.folder, run.record)
            except Exception as error:           # noqa: BLE001 -- the file says what it said before
                self._say_history(run, HISTORY_LEFT_TEXT.format(f"{type(error).__name__}: {error}"))
        return run.record

    @staticmethod
    def _rewrite(folder: Path, run: str, change) -> dict | None:
        """The history of an ended run that the controller no longer holds -- one set aside, or of an earlier launch
        -- read back from its file in ``folder`` and written again with ``change`` made to it alone, as
        ``_write_ended`` makes it. The record now in the file is handed back; None when the file cannot be read
        back as the run's, and then nothing is written. A file that cannot be written is left as it is."""
        record = history.read(folder / f"{run}.json")
        if record is None or record.get("id") != run:
            return None
        changed = change(record) if callable(change) else {**record, **change}
        if changed == record:
            return record
        try:
            history.write(folder, changed)
        except Exception:                        # noqa: BLE001 -- an older run's file that cannot be written stays
            return record
        return changed

    def recall(self, records_root: str) -> None:
        """As a window is opened with no run live and no end waiting to be seen: the newest run of the records folder
        ``records_root`` whose end nobody has seen, as its history file in the folder of runs says it, becomes the
        status, so that the foot and Home say how it ended until it is seen. Nothing when there is none. A history
        file that cannot be read is passed over as though it were not there (``history.read``), and so is one whose
        end cannot be said, which the window's log names."""
        if self._status != IDLE:
            return
        try:
            record = history.newest_unseen(self.folder(), records_root=records_root)
        except OSError:
            record = None
        if record is None or not isinstance(record.get("id"), str):
            return
        try:
            status = ends.status_of(record, self.now())
        except Exception as error:               # noqa: BLE001 -- a history no reading foresaw opens no window's end
            log.warning(history.PASSED_OVER_TEXT, self.folder() / f"{record['id']}.json",
                        f"{type(error).__name__}: {error}")
            return
        self._status = status
        self.status_changed.emit(self._status)

    def show_past(self, run: str, host) -> bool:
        """The run ``run`` on the page of its place, ``host``, which then shows it; whether it was shown. The run the
        controller holds, or one whose page still offers to continue or resume it, on the page it ran on, is shown as
        it stood at its end: a finished run's figures, its result, under "How the run went"; any other its run page,
        with its offer, its question and its failure as they were. Any other run is drawn in the page's past view
        (``past_view``), from its two files alone, its history and its text, beside the run page and never in its
        place, and holding no figures, since the page's own are no past run's: a history that cannot be read, or
        drawn, is said there, and the window's log names its file. Nothing is shown while ``host`` holds a live run
        of its own, whose page stays in sight."""
        held = self._run
        own = held if held is not None and held.id == run else self._offers.get(run)
        if own is not None and own.host is host and _alive(own.view):
            if own.outcome == "finished":
                host._run_over(own.last)
            else:
                host.show_run()
            return True
        if getattr(host, "is_running", lambda: False)():
            return False
        past = getattr(host, "past_view", None)
        if past is None:
            return False
        folder = self.folder()
        path, text = folder / f"{run}.json", folder / f"{run}.txt"
        record = history.read(path)
        try:
            if record is None:
                raise ValueError("it cannot be read")
            past.show_past(record, text if text.is_file() else None, clock=self._clock or _local_now)
        except Exception as error:               # noqa: BLE001 -- a past that cannot be drawn is said on its page
            log.warning(PAST_UNDRAWN_TEXT, path, f"{type(error).__name__}: {error}")
            past.show_unreadable()
        host.show_past_view()
        return True

    def where(self, run: str) -> tuple[str, str | None] | None:
        """The place of the run ``run``: the part and the page that show it, or None when it is not known."""
        held = self._run
        if held is not None and held.id == run:
            return held.spec.place
        folder = self._folder if self._folder is not None else history.runs_folder()
        record = history.read(folder / f"{run}.json")
        place = None if record is None else record.get("place")
        return tuple(place) if isinstance(place, list) and len(place) == 2 else None

    def tick(self) -> None:
        """What the controller's timer does each second while a run is live: the solver's count and the time handed
        to the model, the run's status worked out again, and, at most every ``FSYNC_EVERY_S`` seconds, the run's
        text file asked to reach the disk, on a thread of its own: the disk's time is not the window's. While the run
        waits to be tried again, nothing computes: its page counts down to the next try, which begins at the first
        look its pause has run out by, read on the run's clock."""
        from core.progress import SOLVER
        run = self._run
        if run is None or not self.is_live():
            return
        if run.try_at is not None:
            if (self._clock or _local_now)() >= run.try_at:
                self._begin_try(run)
            else:
                self._draw()
            return
        self._model.solver(SOLVER.steps)
        self._model.tick()
        text = run.text
        now = (self._clock or _local_now)()
        if text is not None and (run.synced is None or (now - run.synced).total_seconds() >= FSYNC_EVERY_S):
            run.synced = now
            threading.Thread(target=text.sync, name="run-text-sync", daemon=True).start()
        self._refresh()
        self._draw()

    # ── what the worker says, on the window's thread ──
    def _on_line(self, text: str, level: str) -> None:
        self._hear([(text, level)])

    def _on_lines(self, batch) -> None:
        self._hear(batch)

    def _hear(self, batch) -> None:
        if self._run is None or not self.is_live() or not batch:
            return
        self._model.feed(batch)
        self.lines.emit(list(batch))
        self._refresh()
        self._draw()

    def _on_rows(self, rows) -> None:
        if self._run is None or not self.is_live():
            return
        self._model.rows(rows)
        self._refresh()
        self._draw()

    # ── the run's page ──
    def _to_page(self, batch) -> None:
        """Lines of the run's full text, handed to the page of the run they belong to: the live run's, or the
        last one's, whose end and history are still being said."""
        view = None if self._run is None else self._run.view
        if _alive(view):
            view.take_lines(batch)

    def _draw(self) -> None:
        """The run's page drawn again from the run's model, as the run's status now says, or, once the run has
        ended, as its status at its end said it, whether or not that end has been seen since; while the run waits to
        be tried again, with the sentence that says what failed and when the next try begins."""
        run = self._run
        view = None if run is None else run.view
        if _alive(view):
            status = self._status if self.is_live() or run.last is None else run.last
            if status.state == "retrying":
                view.show_retrying(retry.failure_words(status, self._model, None)[0])
            view.show_status(status, self._model)

    def _on_result(self, payload) -> None:
        """The run returned: it finished, and the record it wrote, when it hands one back, is noted by its kind, its
        id and its name (never the record itself, which holds what the run computed)."""
        if self._run is not None and self._run.outcome is None:
            self._run.outcome = "finished"
            found = tuple(getattr(payload, part, None) for part in ("kind", "id", "name"))
            self._run.result = found if all(isinstance(part, str) for part in found) and found[0] else None

    def _on_error(self, exc, tb: str, cancel_noted: bool = False, kept_fit=None) -> None:
        run = self._run
        if run is not None:
            run.outcome, run.failure = "failed", (exc, tb, cancel_noted)
            run.kept_fit = run.kept_fit if kept_fit is None else kept_fit      # a try that names none keeps the last

    def _on_cancelled(self, kept_fit=None) -> None:
        run = self._run
        if run is not None:
            run.outcome = "stopped"
            run.kept_fit = run.kept_fit if kept_fit is None else kept_fit

    def _on_finished(self) -> None:
        """The worker's end: how it ended said on ``lines``, after every line it said before it; then, for a
        training that failed or was stopped, the batches its simulation set holds, read from the set now that the
        worker has ended; then, for a failure the window tries again, the run's pause (``_pause``), else the run's
        end (``_end``)."""
        run = self._run
        if run is None or run.worker is None:
            return
        if run.sink.ended is not None:
            self.lines.emit([run.sink.ended])
        failure = run.failure
        run.outcome = run.outcome or "failed"
        refused_early = failure is not None and isinstance(failure[0], Refusal) and not self._spent(run)
        if failure is not None:
            is_oom = _out_of_memory()
            run.failure_kind = ("out of memory" if is_oom(failure[0]) else "device"
                                if retry.is_device_failure(failure[0]) else "other")
        run.kept_now = None
        if run.spec.simulation_set and run.outcome in ("failed", "stopped") and not refused_early:
            batches, complete = self._set_holds(run)
            run.continue_from = batches if batches and not complete else None
            run.kept_now = (batches, int(getattr(run.kept_fit, "passes", 0) or 0))
            if run.outcome == "failed" and self._tried_again(run):
                return
        self._end(run, refused_early)

    def _tried_again(self, run: _Run) -> bool:
        """After a training's failure: whether the window tries it again, and if so the run's pause begun. What the
        run kept now (``kept_now``: the set's batches, the kept training's epochs) is held to what it kept at its last
        failure: a try that kept more starts the count of tries again. Only reopening PRISM can help a failure as
        ``retry.only_reopening`` says, a first failure as well as a try's, and the run then ends; a stop asked for
        before the failure, and any failure not tried again, end it too."""
        exc = run.failure[0]
        is_oom = _out_of_memory()
        kept = run.kept_now
        before, run.last_kept = run.last_kept, kept
        if before is not None and retry.gained(before, kept):
            run.tried = 0
        if retry.only_reopening(exc, kept_before=before or kept, kept_after=kept, is_oom=is_oom, tried=run.tried):
            run.reopen = True
            return False
        if run.stop_requested or not retry.is_tried_again(exc, run.spec, is_oom=is_oom):
            return False
        self._pause(run, kept)
        return True

    def _pause(self, run: _Run, kept: tuple[int, int]) -> None:
        """The run failed and is tried again: the try that failed kept in its history, its worker's payload let go
        (the run's own call is kept, for its next try), and the run waits, live and Retrying, for the pause the tries
        made since it last kept more give it, read on the run's clock; nothing computes meanwhile, its panels stay
        locked, and its text says when it is tried again."""
        now = (self._clock or _local_now)()
        exc = run.failure[0]
        run.attempts.append({"started": run.attempt_started or self._model.started, "ended": now,
                             "outcome": "failed", "failure": type(exc).__name__,
                             "batches_before": run.at_start if run.attempt_before is None else run.attempt_before[0],
                             "batches_after": kept[0],
                             "epochs_before": None if run.attempt_before is None else run.attempt_before[1],
                             "epochs_after": kept[1]})
        run.attempt_open = False
        worker = run.worker
        worker.fn, worker.args, worker.kwargs = None, (), {}
        self._spent_worker, run.worker, run.token, run.outcome = worker, None, None, None
        run.try_at = now + timedelta(seconds=retry.PAUSES_S[run.tried])
        self._model.trying(True)
        self._refresh()
        said = retry.failure_words(self._status, self._model, exc)[0]
        if run.text is not None:
            run.text.line(said, "window")
        self.lines.emit([(said, "window")])
        self._draw()

    def _begin_try(self, run: _Run) -> None:
        """The pause over: the run's next try, its own call in a new worker with a new token, in the same text and
        history files, the text's line that opens it first: the simulation set continued once it holds a batch, or,
        when the run kept a training, that training resumed, as its page's Resume would, and no other argument
        changed. A stop asked for while the run waited whose end of the tries is not taken yet (asked in the turn of
        the event loop in which the pause ran out) ends the tries instead: no try begins."""
        if run.stop_requested:
            self._tries_end()
            return
        run.try_at, run.attempt_open = None, True
        run.tried += 1
        run.attempt_started, run.attempt_before = (self._clock or _local_now)(), run.last_kept
        run.outcome = run.failure = None
        said = SEPARATOR_TEXT.format(len(run.attempts) + 1)
        if run.text is not None:
            run.text.separator(len(run.attempts) + 1)
        self.lines.emit([(said, "window")])
        if run.kept_fit is not None:
            kwargs = retry.resumed_call(run.spec, run.kept_fit.id)
        elif run.last_kept is not None and run.last_kept[0]:
            kwargs = retry.continued_call(run.spec)
        else:
            kwargs = dict(run.spec.kwargs)
        run.worker, run.token = self._worker(run, kwargs)
        self._model.trying(False)
        self._refresh()
        self._draw()
        QThreadPool.globalInstance().start(run.worker)

    def _end(self, run: _Run, refused_early: bool) -> None:
        """The run's end, in this order: its watchers' last look; its state (ended, or idle for a run that leaves
        nothing to acknowledge), with how and when it ended and what it kept in words, read from the history it
        leaves (``ends.words``; a failure's sentence is its page's own); the end that waited to be seen as the run
        started set aside when the run leaves an end of its own, or else the status again (``_set_aside``); the
        history's end written, and kept for every later write, and the text file closed, or both removed for a
        refusal made before anything was spent; every panel unlocked; the computer let sleep again and the reason
        against ending the session withdrawn; the payload let go, and the run's own call with it unless its page
        offers to continue or resume it; the status published; then ``run_ended``; then, for
        an end to be seen, the taskbar's flash on the newest window built; then the run's page drawn ended, a failure
        as its page's state with what the page offers, and its host's column arranged, so that a page that cannot be
        drawn holds up nothing of the run's own end; then the page's own word on a refusal made before anything was
        spent (on the run's page when the run repeats an offer of it), or on the failure of a run that only loads a
        record, which has no page of its own."""
        from ..panels.base_panel import BasePanel
        model = self._model
        for watcher in run.watchers:
            watcher.stop()                       # one last look: the last figure often lands right at the end
            watcher.deleteLater()
        run.watchers = []
        if self._timer is not None:
            self._timer.stop()
        spec, failure = run.spec, run.failure
        model.ended(run.outcome, kept_fit=run.kept_fit)
        run.ended = (self._clock or _local_now)()
        idle = refused_early or spec.loads_only or not spec.keeps_files
        last = RunStatus(state="idle" if idle else "ended", word=spec.word, name=spec.name,
                         run=None if refused_early else run.id, place=spec.place, loading=spec.loads_only,
                         stage=model.stage, outcome=None if refused_early else run.outcome, started=model.started,
                         ended=run.ended, continue_from=run.continue_from, kept_fit=run.kept_fit,
                         closing=run.closing, kinds=spec.kinds, records=spec.records, reopen=run.reopen)
        page = None
        if failure is not None and not refused_early and not spec.loads_only:
            page = self._failure_page(run, last, failure)
        if not idle:
            try:
                sentence, kept = ends.words(self._record(run), run.ended,
                                            too_late=run.stop_requested and run.outcome == "finished")
            except Exception:                    # noqa: BLE001 -- the end's words never hold up the run's end
                sentence, kept = model.sentence(), ""
            last = replace(last, sentence=sentence if page is None else page[0], kept=kept)
        run.last = last
        if spec.word == "Training" and not refused_early:
            self._training_kept = run.continue_from if run.outcome in ("stopped", "failed") else None
        offers = (not refused_early and not idle and spec.simulation_set and run.outcome in ("failed", "stopped")
                  and not run.reopen and (run.continue_from is not None or run.kept_fit is not None)
                  and _alive(run.host))
        self._status = (run.waiting or IDLE) if idle else last
        if not idle:
            self._set_aside(run)
        if spec.keeps_files:
            if refused_early:
                self._remove_files(run)
            else:
                self._write(run)
                run.text.close()
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(True)
        self._stay_awake(False)
        self._withdraw_reason()
        # The payload goes now: the worker's shell outlives the run (it is kept until the next run ends), and what it
        # points at -- the configuration, the prior, the posterior, tensors on the card -- must not. The run's own
        # call goes with it, but for a run whose page offers to continue or resume it: that call is kept until the
        # page starts its next run, or goes (its panel's deletion lets the call go at once).
        worker = run.worker
        if worker is not None:
            worker.fn, worker.args, worker.kwargs = None, (), {}
            self._spent_worker = worker
        if offers:
            self._offers[run.id] = run
            run.host.destroyed.connect(partial(self._page_gone, run.id))
        else:
            spec = replace(spec, fn=None, args=(), kwargs={})
            run.spec = model.spec = spec
            run.keywords = {}
        run.worker, run.token, run.failure = None, None, None
        self.status_changed.emit(self._status)
        self.run_ended.emit(last)
        if not idle:
            this_computer.flash(self._the_window())
        if _alive(run.view):
            if page is not None:
                run.view.show_failure(*page)
            run.view.offer(run.continue_from if offers else None, offers and run.kept_fit is not None)
            run.view.show_status(last, model)
        host = run.host
        said_on_the_page = refused_early and run.repeats is not None and _alive(run.view)
        if said_on_the_page:
            run.view.refused(self._refusal_words(failure[0]))
        elif _alive(host) and hasattr(host, "_run_over"):
            host._run_over(last)
        if failure is not None and _alive(host) and not said_on_the_page:
            if refused_early:
                host._refusal(failure[0])
            elif spec.loads_only:
                host._on_error(failure[0], failure[1], failure[2], run.kept_fit)

    @staticmethod
    def _refusal_words(exc) -> str:
        """A refusal as a page says it: the stage's sentence, then where to fix it, when the window names a
        place."""
        from .. import fields as gui_fields
        return f"{exc.message} {gui_fields.fix_sentence(exc.field)}".rstrip()

    def _failure_page(self, run: _Run, last: RunStatus, failure) -> tuple:
        """What the page of a run that failed shows: its sentence, then, as ``show_failure`` takes them, the error's
        message (a refusal's with where to fix it, as the window says a refusal), its kind and traceback, whether a
        stop had been asked for before it, whether the run keeps a file that holds them, and the command line the
        page sets off beneath its sentence ("" for none)."""
        exc, trace, noted = failure
        sentence, _button = retry.failure_words(last, self._model, exc, is_oom=_out_of_memory())
        said = retry.message(exc)
        quoted = self._refusal_words(exc) if isinstance(exc, Refusal) else said
        return (sentence, quoted, trace or f"{type(exc).__name__}: {said}", bool(noted), run.text is not None,
                retry.reopen_command(last, self._model))

    def _set_holds(self, run: _Run) -> tuple[int, bool]:
        """The batches the run's simulation set holds, and whether it is complete, read from the set itself, in the
        folder its run named: once the run's worker has ended and never while it runs, since the engine replaces
        what is read here at each save. No batch, and not complete, when the run named no set or none can be read
        there."""
        folder = self._model.simulation_folder()
        if folder is None:
            return 0, False
        try:
            from core.SBI.training_checkpoint import peek
            state = peek(run.store.kind_dir("simulation") / folder) or {}
            done, complete = int(state.get("batches_done") or 0), bool(state.get("complete"))
        except Exception:                        # noqa: BLE001 -- a set that cannot be read holds nothing to offer
            return 0, False
        return max(done, 0), complete

    @staticmethod
    def _remove_files(run: _Run) -> None:
        """The run's two files removed, its text file closed first, and its history written no more. A file that
        cannot be removed now (one another program holds open, which Windows will not let go of) stays."""
        run.writes = False
        run.text.close()
        for path in (run.folder / f"{run.id}.json", run.folder / f"{run.id}.txt"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _spent(self, run: _Run) -> bool:
        """Whether something was spent since the start: the solver took a step, a stage after the run's first has
        begun, or the store holds a record of a kind the run writes made since the start. A store that cannot be
        listed counts as spent: the run's files are then kept."""
        from core.artifacts import StoreError
        from core.progress import SOLVER
        if moved(run.solver_at_start, SOLVER.steps) or self._model.began_after_first():
            return True
        for kind in run.spec.kinds:
            try:
                listed = run.store.list(kind)
            except (OSError, StoreError):
                return True
            for summary in listed:
                made = _made(summary.created)
                if made is not None and made >= run.real_start:
                    return True
        return False

    # ── the run's state, and its history ──
    def _live_status(self) -> RunStatus:
        run, model, spec = self._run, self._model, self._run.spec
        waits = run.try_at is not None
        state = ("retrying" if waits else model.state if model.state in ("starting", "running", "stopping")
                 else "running")
        stage = model.stage
        words = questions.stopping_line(self._status, model) if state == "stopping" else model.doing()
        return RunStatus(state=state, word=spec.word, name=spec.name, run=run.id, place=spec.place,
                         loading=spec.loads_only,
                         state_word=("loading" if spec.loads_only else "stopping" if state == "stopping"
                                     else "trying again" if waits else "running"),
                         stage=stage, stage_words=words, needs_you=model.needs_you(), started=model.started,
                         continue_from=run.continue_from if waits else None,
                         kept_fit=run.kept_fit if waits else None, stop_requested=run.stop_requested,
                         closing=run.closing, kinds=spec.kinds, records=spec.records,
                         trying=run.tried + 1 if waits else 0, try_at=run.try_at)

    def _refresh(self) -> None:
        """The run's state worked out again after the model heard something: the history written at a change of
        state or stage, at a save of the simulation set passed and at an epoch; the status published at a change of
        it and at a save passed, and once it is published the taskbar flashed when the run has come to need
        someone."""
        run, model = self._run, self._model
        named = not run.set_named and model.simulation_folder() is not None
        if named:
            self._name_the_set(run)
        status = self._live_status()
        saves, epochs = model.saves_passed(), len(model.epochs())
        saved = saves != run.saves
        if (named or saved or epochs != run.epochs or status.state != self._status.state
                or status.stage != self._status.stage):
            run.saves, run.epochs = saves, epochs
            self._write(run)
        if saved or status != self._status:
            needed = self._status.needs_you
            self._status = status
            self.status_changed.emit(status)
            if status.needs_you and not needed:
                this_computer.flash(self._the_window())
        self._give_reason()

    # ── this computer, while a run is live ──
    def _stay_awake(self, on: bool) -> None:
        """Ask Windows to keep the computer awake, or let it sleep again: once each way."""
        if on != self._awake:
            self._awake = on
            this_computer.keep_awake(on)

    def _the_window(self):
        """The newest window built, while it is there; None otherwise."""
        window = None if self._window is None else self._window()
        return window if _alive(window) else None

    def _give_reason(self) -> None:
        """The reason against ending the session, registered once the live run is past its start, and again at each
        change of its stage or of the words for it."""
        status = self._status
        if not status.live or status.state == "starting":
            return
        reason = (status.stage, questions.shutdown_reason(status, self._model))
        if reason != self._reason:
            self._reason = reason
            this_computer.block_shutdown(self._the_window(), reason[1])

    def _withdraw_reason(self) -> None:
        """The reason against ending the session withdrawn, when one is registered."""
        if self._reason is not None:
            self._reason = None
            this_computer.block_shutdown(self._the_window(), None)

    def _name_the_set(self, run: _Run) -> None:
        """The simulation set's folder is named: the batches it held at the start are noted, and the model is
        handed the failed and simulated counts of the set's earlier runs."""
        run.set_named = True
        folder = self._model.simulation_folder()
        run.at_start = self._model.batches_done() or 0
        try:
            earlier = [record for record in history.runs_for(run.folder, folder) if record.get("id") != run.id]
        except OSError:
            earlier = []
        counts = []
        for record in earlier:
            health = record.get("health") or {}
            failed, made = health.get("failed"), health.get("simulated")
            if all(isinstance(n, int) and not isinstance(n, bool) for n in (failed, made)):
                counts.append((failed, made))
        if counts:
            self._model.set_earlier(tuple(counts))

    def _write(self, run: _Run) -> None:
        """Write the run's history file whole, while it is still written; at the run's end, the history written is
        kept, and every later write starts from it. A write that fails is tried again at the run's next moment that
        writes it, and the run goes on; the first failure is said, once, in the run's full text and its text file."""
        if not run.writes:
            return
        try:
            record = self._record(run)
            if run.ended is not None:
                run.record = record              # the end's: every later write starts from it (``_write_ended``)
            history.write(run.folder, record)
        except Exception as error:               # noqa: BLE001 -- the history never stops a run
            self._say_history(run, HISTORY_BROKEN_TEXT.format(f"{type(error).__name__}: {error}"))

    def _say_history(self, run: _Run, said: str) -> None:
        """A history file that could not be written, ``said`` in the run's full text and its text file, once for the
        run: its text file, closed at the run's end, takes nothing more after it."""
        if run.history_said:
            return
        run.history_said = True
        if run.text is not None:
            run.text.line(said, "window")
        self.lines.emit([(said, "window")])

    def _record(self, run: _Run) -> dict:
        """The run's history, as its model reads it now, worked out until the run's end and never after it: that its
        end was seen, and that the training it kept was discarded, are written once it has ended, into the history its
        end wrote (``_write_ended``). A stop asked with a reason of its own makes the run's outcome
        "stopped", with that reason, until the run has ended, and its end the moment the reason was said, and the same
        of the try it is in, with the moment the run was last heard, at each write: a process Windows ends before the
        run leaves them said, and leaves said that the run's own end was never written. The reason stays unless the
        run finished. The record's name asked is the name the run's call was handed, never the run's own word for
        itself."""
        spec, model = run.spec, self._model
        reason = run.stop_reason if run.stop_reason and run.outcome != "finished" else None
        ended = run.ended if run.ended is not None or reason is None else run.reason_at
        epochs, budget, failed = model.epochs(), model.budget() or {}, model.failed()
        folder, note = model.simulation_folder(), run.kept_fit
        network = spec.facts.get("network")
        fit = None
        if epochs or note is not None:
            best = model.best_epoch()
            loss = next((record.validation_loss for record in epochs if record.epoch == best), None)
            paces = [record.seconds for record in epochs[-FIT_PACE_RECORDS:]]
            stopped = note is not None and (run.outcome in ("stopped", "failed") or run.try_at is not None)
            fit = {"epochs": note.passes if stopped else (epochs[-1].epoch if epochs else None),
                   "best_epoch": best,
                   "best_validation_loss": loss if loss is not None and math.isfinite(loss) else None,
                   "seconds_per_epoch": float(statistics.median(paces)) if paces else None,
                   "ended_by": model.fit_ended_by() if run.outcome == "finished" else None,
                   "network": list(network) if network else None, "simulations": budget.get("simulations"),
                   "kept_id": getattr(note, "id", None), "passes": getattr(note, "passes", None),
                   "resumed": getattr(note, "resumed", None), "processes": getattr(note, "processes", None),
                   "wall_seconds": None, "save_error": None, "expected": model.expected_epochs(),
                   "discarded": False}
        failure = None
        if run.failure is not None and run.outcome == "failed":
            exc = run.failure[0]
            try:
                message = str(exc)
            except Exception:                    # noqa: BLE001 -- the failure is kept by its kind alone
                message = ""
            failure = {"type": type(exc).__name__, "message": message, "kind": run.failure_kind,
                       "reopen": run.reopen}
        done = model.batches_done()
        tries = list(run.attempts)               # each try that failed and was tried again
        if run.attempt_open:                     # the try running, or the one the run ended with
            before = run.attempt_before
            # Once the run has ended, what the try left is what the set and the kept training hold, as for a try that
            # failed and was tried again; while it runs, what its text has counted.
            after = run.kept_now if run.ended is not None else None
            tries.append({"started": run.attempt_started or model.started, "ended": ended,
                          "outcome": run.outcome or ("stopped" if reason else None),
                          "failure": None if failure is None else failure["type"],
                          "batches_before": run.at_start if before is None else before[0],
                          "batches_after": done if after is None else after[0],
                          "epochs_before": None if before is None else before[1],
                          "epochs_after": (epochs[-1].epoch if epochs else None) if after is None else after[1]})
        return {"id": run.id, "word": spec.word, "name": spec.name,
                "place": None if spec.place is None else list(spec.place),
                "records_root": str(getattr(run.store, "root", "")), "started": model.started,
                "ended": ended, "last_heard": (self._clock or _local_now)() if reason and run.ended is None else None,
                "outcome": run.outcome or ("stopped" if reason else "running"), "stop_reason": reason,
                "stage": model.stage,
                "stages": [[stage.name, list(stage.begins), stage.counts] for stage in spec.stages],
                "acknowledged": False, "tries": tries,
                "plan": None, "settings": None, "memory_limit": spec.facts.get("memory_limit"),
                "simulation_set": None if folder is None else {
                    "folder": folder, "batches": budget.get("batches"), "at_start": run.at_start, "at_end": done,
                    "per_batch": budget.get("rows"), "seconds_per_batch": model.seconds_per_batch()},
                "health": None if failed is None else {"failed": failed[0], "simulated": failed[1]},
                "fit": fit, "check": None,
                "result": None if run.result is None else dict(zip(("kind", "id", "name"), run.result),
                                                               asked=run.asked),
                "failure": failure, "text": f"{run.id}.txt"}


def _discarded(record: dict) -> dict:
    """A run's history, its kept training said to be discarded; as it was when it holds no fit."""
    fit = record.get("fit")
    return {**record, "fit": {**fit, "discarded": True}} if isinstance(fit, dict) else record


def _said_again(status: RunStatus, record: dict, when: datetime) -> RunStatus:
    """``status``, an end of the run whose history is ``record``, with what it kept said again from that history, and
    how it ended too but for a failure, whose sentence is its page's own beside its error; dated against ``when``. As
    it was when that history's words cannot be read."""
    try:
        sentence, kept = ends.words(record, when)
    except Exception:                            # noqa: BLE001 -- the words stay as the end said them
        return status
    return replace(status, kept=kept, sentence=status.sentence if status.outcome == "failed" else sentence)


def _alive(widget) -> bool:
    """Whether ``widget`` is there to be told: given, and not deleted."""
    if widget is None:
        return False
    import shiboken6
    return shiboken6.isValid(widget)


_ONE: RunController | None = None


def controller() -> RunController:
    """The process's one run controller, made at the first call."""
    global _ONE
    if _ONE is None:
        _ONE = RunController()
    return _ONE
