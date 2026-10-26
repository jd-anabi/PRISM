"""The run's controller: the one run of the process, from the start a page asks for to the end it leaves behind.

There is one per process (``controller()``), because a run's output is captured for the whole process and only one
run goes at a time; every window hears it, each through methods of its own, so that a window deleted leaves no
connection behind. A page starts a run with ``dispatch``: the controller writes the run's history file and its text
file (where the run keeps them), makes the run's model, locks the controls of every panel, says the run is starting
and queues the worker. All the worker says reaches the controller on the window's thread, where the model is fed,
the run's state worked out again, the history written (at each change of state or stage, at each save of the
simulation set the run passes, at each epoch and at the end) and every line handed on to the run's full text
(``lines``). At the end the panels are unlocked and the payload let go, the run's status is published last of all,
then ``run_ended``; only then does the page show how it ended, when that takes a box.

A refusal the stage makes before anything is spent is no run: the solver has taken no step since the start, no
stage after the run's first has begun, and the store holds no record of a kind the run writes made since the start.
Its two files are removed, the controller is idle again with nothing to acknowledge, and the page shows the refusal
as it shows one made at the click.

A run that only loads a record, a live simulation and a video export keep no file and end straight into idle, with
nothing to acknowledge.

Nothing here draws: a figure a run hands over goes to its page's stack of pictures as the picture it already is.
"""
import json
import math
import os
import statistics
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal

from core.refusals import Refusal

from ..streams import CancelToken
from ..worker import Worker
from . import history
from .model import FIT_PACE_RECORDS, IDLE, RunModel, RunSpec, RunStatus, moved
from .textfile import FSYNC_EVERY_S, RunTextFile

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
HISTORY_BROKEN_TEXT = "The run's history file could not be written ({}); the run goes on without it."


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


def _made(created) -> datetime | None:
    """A record's moment as its manifest gives it, or None when it cannot be read as one with its zone."""
    try:
        moment = datetime.fromisoformat(created)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else None


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
    """What the worker writes as the run's text: the run's text file, when the run keeps one, and the run's full
    text in the window. ``line`` is the run's line sink, called under the stream pump's lock: it only writes the
    file (the window hears the same line through the worker's own signals). ``outcome``, how the run ended, is
    written to the file and said on the controller's ``lines`` too, from the worker's thread."""

    def __init__(self, file: RunTextFile | None, say):
        self._file = file
        self._say = say

    def line(self, text: str, level: str) -> str | None:
        return None if self._file is None else self._file.line(text, level)

    def outcome(self, text: str, level: str, detail: str = "") -> None:
        if self._file is not None:
            self._file.outcome(text, level, detail)
        self._say([(f"{text}\n{detail}" if detail else text, level)])


class _Run:
    """What the controller holds of one run: the live one, and the last one until the next starts."""

    def __init__(self, spec: RunSpec, host, store, folder: Path):
        self.spec, self.host, self.store, self.folder = spec, host, store, folder
        self.id: str | None = None                 # its files' stem; None for a run that keeps none
        self.text: RunTextFile | None = None
        self.writes = spec.keeps_files             # whether its history file is still written
        self.worker = None
        self.token: CancelToken | None = None
        self.watchers: list = []
        self.solver_at_start = 0
        self.real_start: datetime | None = None    # the computer's own time at the start, for the store's records
        self.outcome: str | None = None            # "finished", "stopped" or "failed"
        self.failure = None                        # (exception, traceback, cancel noted) of a failed run
        self.kept_fit = None
        self.stop_requested = False
        self.ended: datetime | None = None
        self.acknowledged = False
        self.set_named = False
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
    # A stop asked for, heard on the window's thread, from whichever thread asked.
    _stopping = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stopping.connect(self._stop_heard)
        self._status: RunStatus = IDLE
        self._model: RunModel | None = None
        self._run: _Run | None = None
        self._clock = None                         # None: the computer's own clock
        self._folder: Path | None = None           # None: the folder of runs beside the settings file
        self._timer: QTimer | None = None          # made at the first run: nothing ticks before it
        self._spent_worker = None                  # the last run's worker, kept until the next run ends

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

    def use_clock(self, clock) -> None:
        """The clock every moment of the next runs is read from, local time with its zone; None for the
        computer's own."""
        self._clock = clock

    def use_folder(self, folder) -> None:
        """The folder of runs the next runs keep their files in, and their paces and earlier runs are read from;
        None for the one beside the settings file in use when each starts."""
        self._folder = None if folder is None else Path(folder)

    # ── a run's start ──
    def dispatch(self, fn, *args, spec: RunSpec, host=None, on_result=None, provide_fig_sink: bool = False,
                 provide_stream: bool = False, on_chunk=None, watch_dir=None, **kwargs) -> bool:
        """Start ``fn(*args, **kwargs)`` as the process's run, described by ``spec``, for the page ``host``. False,
        and nothing started, while a run is live. Else, in this order: the spec is given the call; the memory limit
        in force is read once; where the run keeps files, its id, its history file and its text file with its head;
        its model; the controls of every panel locked; the status published; the worker queued. ``on_result`` hears
        the run's result while the run is still live; ``provide_fig_sink`` hands the call a sink for its figures,
        which go to the host's stack of pictures; ``provide_stream`` hands it ``emit_chunk`` and ``should_stop``,
        and ``on_chunk`` hears each chunk; ``watch_dir`` is one folder, or several, whose new pictures go to the
        host's stack. None of these is handed on to ``fn``."""
        if self.is_live():
            return False
        from core.artifacts import default_store
        from core.progress import SOLVER
        from ..panels.base_panel import BasePanel, _png_fig_sink
        from ..plot_watcher import NewPngWatcher
        clock = self._clock or _local_now
        gib, limit_text = _memory_limit()
        spec = spec.with_call(fn, args, kwargs)
        spec = replace(spec, facts={**spec.facts, "memory_limit": gib})
        store = default_store()
        folder = self._folder if self._folder is not None else history.runs_folder()
        run = _Run(spec, host, store, folder)
        run.solver_at_start, run.real_start = SOLVER.steps, datetime.now(timezone.utc)
        expected = _expected(store, spec.facts) if "model" in spec.facts else None
        word = spec.word
        model = RunModel(spec, clock=clock, expected=expected,
                         estimate=lambda per, size: history.estimate(folder, word=word, per=per, size=size))
        self._model, self._run = model, run
        said = []
        if spec.keeps_files:
            run.id = history.run_id(spec.word, model.started, folder)
            self._write(run)
            run.text = RunTextFile(folder / f"{run.id}.txt", clock=clock)
            said = [(f"{spec.word} {spec.name}, run {run.id}." if spec.name else f"{spec.word}, run {run.id}.",
                     "window"), (limit_text, "window")]
            broke = run.text.head(word=spec.word, name=spec.name, run=run.id, memory_limit=limit_text)
            if broke is not None:
                said.append((broke, "warning"))
                model.feed([(broke, "warning")])
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(False)
        if host is not None and hasattr(host, "_hosting"):
            host._hosting(True)
        self._status = self._live_status()
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(TICK_MS)
            self._timer.timeout.connect(self.tick)
        self._timer.start()
        if said:
            self.lines.emit(said)
        self.status_changed.emit(self._status)

        token = CancelToken()
        worker = Worker(fn, *args, cancel=token, run_text=_Text(run.text, self.lines.emit), **kwargs)
        # Kept until the run has ended and its signals have been heard, and never deleted by the pool: a worker the
        # pool deleted as its run returned would take its still-queued signals with it.
        worker.setAutoDelete(False)
        signals = worker.signals
        if provide_fig_sink:
            worker.kwargs["fig_sink"] = _png_fig_sink(signals.figure)
        if provide_stream:
            # A long-lived streaming run hands each chunk to ``emit_chunk`` and polls ``should_stop`` to unwind.
            worker.kwargs["emit_chunk"] = signals.chunk.emit
            worker.kwargs["should_stop"] = token.requested.is_set
            if on_chunk is not None:
                signals.chunk.connect(on_chunk)
        stack = getattr(host, "figure_stack", None)
        if stack is not None:
            signals.figure.connect(stack.add_figure)
        signals.log.connect(self._on_line)
        signals.log_batch.connect(self._on_lines)
        signals.rows.connect(self._on_rows)
        if on_result is not None:
            signals.result.connect(on_result)
        signals.result.connect(self._on_result)
        signals.error.connect(self._on_error)
        signals.cancelled.connect(self._on_cancelled)
        signals.finished.connect(self._on_finished)
        # One watcher for each folder: a watcher reads one folder and not the folders under it, and a sweep study
        # writes two records. The controller holds them, so a page deleted during the run takes none with it.
        for one in ([] if watch_dir is None else
                    [watch_dir] if isinstance(watch_dir, (str, Path)) else list(watch_dir)):
            watcher = NewPngWatcher(one, self)
            if stack is not None:
                watcher.png_ready.connect(stack.add_png)
            watcher.start()
            run.watchers.append(watcher)
        run.worker, run.token = worker, token
        QThreadPool.globalInstance().start(worker)
        return True

    # ── a run's stop and its end ──
    def request_stop(self) -> None:
        """Ask the live run to stop at its next point where a stop is taken: the run is Stopping from now on.
        Nothing when no run is live, or a stop is asked for already. Safe from any thread: the run's token is set
        at once, and the run's state and history follow on the window's thread (at once when asked there)."""
        run = self._run
        token = None if run is None else run.token
        if token is None or not self.is_live() or run.stop_requested:
            return
        run.stop_requested = True
        token.requested.set()
        self._stopping.emit()

    def _stop_heard(self) -> None:
        """A stop asked for, on the window's thread: the model is told, and the run's state worked out again."""
        run = self._run
        if run is not None and run.stop_requested and self.is_live():
            self._model.stopping(True)
            self._refresh()

    def acknowledge(self, run: str) -> None:
        """Someone has seen how the run ``run`` ended: its history file says so, and when it is the run the
        controller holds, the controller is idle again."""
        held = self._run
        if held is not None and held.id == run and self._status.unacknowledged_end:
            held.acknowledged = True
            self._write(held)
            self._status = IDLE
            self.status_changed.emit(IDLE)
            return
        folder = self._folder if self._folder is not None else history.runs_folder()
        record = history.read(folder / f"{run}.json")
        if record is not None and record.get("id") == run and not record.get("acknowledged"):
            try:
                history.write(folder, {**record, "acknowledged": True})
            except Exception:                    # noqa: BLE001 -- an older run's file that cannot be written stays
                pass

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
        text file asked to reach the disk, on a thread of its own: the disk's time is not the window's."""
        from core.progress import SOLVER
        run = self._run
        if run is None or not self.is_live():
            return
        self._model.solver(SOLVER.steps)
        self._model.tick()
        text = run.text
        now = (self._clock or _local_now)()
        if text is not None and (run.synced is None or (now - run.synced).total_seconds() >= FSYNC_EVERY_S):
            run.synced = now
            threading.Thread(target=text.sync, name="run-text-sync", daemon=True).start()
        self._refresh()

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

    def _on_rows(self, rows) -> None:
        if self._run is None or not self.is_live():
            return
        self._model.rows(rows)
        self._refresh()

    def _on_result(self, _payload) -> None:
        if self._run is not None and self._run.outcome is None:
            self._run.outcome = "finished"

    def _on_error(self, exc, tb: str, cancel_noted: bool = False, kept_fit=None) -> None:
        run = self._run
        if run is not None:
            run.outcome, run.failure, run.kept_fit = "failed", (exc, tb, cancel_noted), kept_fit

    def _on_cancelled(self, kept_fit=None) -> None:
        run = self._run
        if run is None:
            return
        run.outcome, run.kept_fit = "stopped", kept_fit
        host = run.host
        if _alive(host) and hasattr(host, "log_pane"):
            host.log_pane.append_line("Run cancelled.", "warning")

    def _on_finished(self) -> None:
        """The run's end, in this order: its watchers' last look; its state (ended, or idle for a run that leaves
        nothing to acknowledge); the history's end written and the text file closed, or both removed for a refusal
        made before anything was spent; every panel unlocked; the payload let go; the status published; then
        ``run_ended``; then the page's own word on a failure or a refusal."""
        from ..panels.base_panel import BasePanel
        run, model = self._run, self._model
        if run is None or run.worker is None:
            return
        for watcher in run.watchers:
            watcher.stop()                       # one last look: the last figure often lands right at the end
            watcher.deleteLater()
        run.watchers = []
        if self._timer is not None:
            self._timer.stop()
        spec, failure = run.spec, run.failure
        run.outcome = run.outcome or "failed"
        refused_early = failure is not None and isinstance(failure[0], Refusal) and not self._spent(run)
        model.ended(run.outcome, kept_fit=run.kept_fit)
        run.ended = (self._clock or _local_now)()
        idle = refused_early or spec.loads_only or not spec.keeps_files
        last = RunStatus(state="idle" if idle else "ended", word=spec.word, name=spec.name,
                         run=None if refused_early else run.id, place=spec.place, loading=spec.loads_only,
                         stage=model.stage, outcome=None if refused_early else run.outcome, started=model.started,
                         ended=run.ended, kept_fit=run.kept_fit, kinds=spec.kinds, records=spec.records)
        self._status = IDLE if idle else last
        if spec.keeps_files:
            if refused_early:
                run.writes = False
                run.text.close()
                for path in (run.folder / f"{run.id}.json", run.folder / f"{run.id}.txt"):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
            else:
                self._write(run)
                run.text.close()
        for panel in list(BasePanel._instances):
            panel.set_controls_enabled(True)
        host = run.host
        if _alive(host) and hasattr(host, "_hosting"):
            host._hosting(False)
        # The payload goes now: the worker's shell outlives the run (it is kept until the next run ends), and what it
        # points at -- the configuration, the prior, the posterior, tensors on the card -- must not.
        worker = run.worker
        worker.fn, worker.args, worker.kwargs = None, (), {}
        spec = replace(spec, fn=None, args=(), kwargs={})
        run.spec = model.spec = spec
        self._spent_worker, run.worker, run.token, run.failure = worker, None, None, None
        self.status_changed.emit(self._status)
        self.run_ended.emit(last)
        if failure is not None and _alive(host):
            if refused_early:
                host._refusal(failure[0])
            else:
                host._on_error(failure[0], failure[1], failure[2], run.kept_fit)

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
        state = model.state if model.state in ("starting", "running", "stopping") else "running"
        stage = model.stage
        words = next((word.word for word in model.stage_line() if word.name == stage), "")
        return RunStatus(state=state, word=spec.word, name=spec.name, run=run.id, place=spec.place,
                         loading=spec.loads_only,
                         state_word=("loading" if spec.loads_only else "stopping" if state == "stopping"
                                     else "running"),
                         stage=stage, stage_words=words, needs_you=model.needs_you(), started=model.started,
                         stop_requested=run.stop_requested, kinds=spec.kinds, records=spec.records)

    def _refresh(self) -> None:
        """The run's state worked out again after the model heard something: the history written at a change of
        state or stage, at a save of the simulation set passed and at an epoch; the status published at a change of
        it and at a save passed."""
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
            self._status = status
            self.status_changed.emit(status)

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
        """Write the run's history file whole, while it is still written. One that cannot be written is not written
        again, and the run's full text and its text file say so; the run goes on."""
        if not run.writes:
            return
        try:
            history.write(run.folder, self._record(run))
        except Exception as error:               # noqa: BLE001 -- the history never stops a run
            run.writes = False
            said = HISTORY_BROKEN_TEXT.format(f"{type(error).__name__}: {error}")
            if run.text is not None:
                run.text.line(said, "window")
            self.lines.emit([(said, "window")])

    def _record(self, run: _Run) -> dict:
        """The run's history, as its model reads it now."""
        spec, model = run.spec, self._model
        epochs, budget, failed = model.epochs(), model.budget() or {}, model.failed()
        folder, note = model.simulation_folder(), run.kept_fit
        network = spec.facts.get("network")
        fit = None
        if epochs or note is not None:
            best = model.best_epoch()
            loss = next((record.validation_loss for record in epochs if record.epoch == best), None)
            paces = [record.seconds for record in epochs[-FIT_PACE_RECORDS:]]
            stopped = note is not None and run.outcome in ("stopped", "failed")
            fit = {"epochs": note.passes if stopped else (epochs[-1].epoch if epochs else None),
                   "best_epoch": best,
                   "best_validation_loss": loss if loss is not None and math.isfinite(loss) else None,
                   "seconds_per_epoch": float(statistics.median(paces)) if paces else None, "ended_by": None,
                   "network": list(network) if network else None, "simulations": budget.get("simulations"),
                   "kept_id": getattr(note, "id", None), "passes": getattr(note, "passes", None),
                   "resumed": getattr(note, "resumed", None), "processes": getattr(note, "processes", None),
                   "wall_seconds": None, "save_error": None}
        failure = None
        if run.failure is not None:
            exc = run.failure[0]
            try:
                message = str(exc)
            except Exception:                    # noqa: BLE001 -- the failure is kept by its kind alone
                message = ""
            failure = {"type": type(exc).__name__, "message": message}
        done = model.batches_done()
        return {"id": run.id, "word": spec.word, "name": spec.name,
                "place": None if spec.place is None else list(spec.place),
                "records_root": str(getattr(run.store, "root", "")), "started": model.started, "ended": run.ended,
                "last_heard": None, "outcome": run.outcome or "running", "stage": model.stage,
                "acknowledged": run.acknowledged,
                "tries": [{"started": model.started, "ended": run.ended, "outcome": run.outcome,
                           "failure": None if failure is None else failure["type"], "batches_before": run.at_start,
                           "batches_after": done, "epochs_before": None,
                           "epochs_after": epochs[-1].epoch if epochs else None}],
                "plan": None, "settings": None, "memory_limit": spec.facts.get("memory_limit"),
                "simulation_set": None if folder is None else {
                    "folder": folder, "batches": budget.get("batches"), "at_start": run.at_start, "at_end": done,
                    "per_batch": budget.get("rows"), "seconds_per_batch": model.seconds_per_batch()},
                "health": None if failed is None else {"failed": failed[0], "simulated": failed[1]},
                "fit": fit, "check": None, "result": None, "failure": failure, "text": f"{run.id}.txt"}


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
