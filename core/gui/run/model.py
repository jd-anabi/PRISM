"""A run as the window reads it: its state, its sentence, its stage line, when it should end, its health and its
messages, all read from the lines and the progress rows the run writes, with no window and no thread.

The model is fed what the run's streams carry (``feed`` with its lines, ``rows`` with its progress rows) and the
solver's step count (``solver``), and is asked to look at the time (``tick``); every moment it holds is read from
the clock it is handed, which gives local time with its zone, so nothing here waits and a test drives a run of days
in a moment. It imports neither the toolkit nor the engine nor the engine's settings: a progress row is read by its
attributes, and the solver's count is handed in.

What the page knows and the run does not (its word and name, its stages, the patience and the epoch limit it was
started with) is the run's ``RunSpec``; what the shell reads of a run is its ``RunStatus``.
"""
import math
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from ..texts import date_text, duration_text, number_text
from . import readings

#: The batches the simulation's pace is measured over, the newest of this try.
PACE_BATCHES = 200
#: The batches a simulation times before its own pace is taken over the history's estimate.
TIMED_BATCHES = 20
#: Any other count is timed over its total divided by SHORT_COUNT_PARTS when that is fewer than TIMED_BATCHES, and
#: over SHORT_COUNT_LEAST steps at the least: a study of eight points gets its end time from its third.
SHORT_COUNT_PARTS = 4
SHORT_COUNT_LEAST = 2
#: The epochs whose median time is the fit's pace.
FIT_PACE_RECORDS = 10
#: Seconds with no line and no step of the solver after which the run says so.
SILENCE_ATTENTION_S = 45.0
#: Seconds of such a silence, outside the fit, after which the run needs someone.
SILENCE_NEEDS_YOU_S = 600.0
#: The messages the page keeps, the newest.
MESSAGES_KEPT = 50
#: The share of failed simulations up to which their count reads "Fine".
FAILED_FINE = 0.01

STATES = ("idle", "starting", "running", "stopping", "retrying", "ended", "not_seen", "acknowledged")
OUTCOMES = ("finished", "stopped", "failed", "not seen to end")
#: The progress row that counts a training's simulation.
SIMULATION_ROW = "Generating training data"
#: What the model itself says when a record of the fit shows that the fit has ended: its count since the best
#: has reached the patience, or its epoch is the limit plus one. A stage may begin with it.
FIT_ENDED = "fit_ended"
#: The readings that leave out the stage in progress: it is not drawn on the stage line.
LEAVES_OUT = ("fdt_sanity_skipped",)
#: The opening word of a message of each kind.
LEADS = {"warn": "Attention", "err": "Error", "info": "Information"}

_ENDED_WORDS = {"stopped": "Stopped", "failed": "Failed", "not seen to end": "Not seen to end"}
_ENDED_SENTENCES = {"finished": "{who} finished.", "stopped": "{who} was stopped.", "failed": "{who} failed.",
                    "not seen to end": "{who} was not seen to end."}
_ROTATION_WORDS = {"reused": "Reused from the saved simulation set", "parent": "The parent posterior's"}
_COUNTER = re.compile(r"Training neural network\. Epochs trained: (\d+)")
_SINCE = re.compile(r"(\d+) passes since the best")      # in a resumed training's line, when a loss was finite
_DONE_OF = re.compile(r"\s*(\d+)/(\d+)")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")
_OWN = "own"        # the samples of a run counted by its driving row or by set_count, not by a stage


@dataclass(frozen=True)
class Stage:
    """One stage of a run: its name ("Simulation"), the readings that begin it (none: it begins with the run), and
    what counts it, a progress row's description or a reading's name ("Generating training data", "fit_epoch")."""
    name: str
    begins: tuple[str, ...] = ()
    counts: str | None = None


@dataclass(frozen=True)
class RunSpec:
    """What a page knows of the run it starts and the run does not say: its word and name, its place, its stages,
    whether it only loads a record, what it writes, whether it has a simulation set to continue, the patience, the
    epoch limit and the expected epochs it was started with, how it stops, and the call itself, for a try again.
    ``facts`` are what a pace and the expected epochs are looked up by ("model", "observation": the kind of
    observation, "narrowed", "network": hidden features and transforms), and what the page knows before the run
    says it ("rotated": False for a training with no rotation; "memory_limit": the graphics card's limit in GiB;
    "cadence": the batches between two saves of a simulation set the run continues)."""
    word: str
    place: tuple[str, str | None]
    name: str = ""
    stop_button: str = ""
    stages: tuple[Stage, ...] = ()
    loads_only: bool = False
    keeps_files: bool = True
    kinds: tuple[str, ...] = ()
    records: tuple[str, ...] = ()
    simulation_set: bool = False
    patience: int | None = None
    epoch_limit: int | None = None
    expected_epochs: int | None = None
    retries: bool = False
    stop_asks: bool = True
    own_view: bool = False
    facts: dict = field(default_factory=dict)
    fn: Callable | None = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def with_call(self, fn, args: tuple, kwargs: dict) -> "RunSpec":
        """The same run, with the call that starts it."""
        return replace(self, fn=fn, args=tuple(args), kwargs=dict(kwargs))


@dataclass(frozen=True)
class RunStatus:
    """What the shell reads of the run: its state and the words for it, where it is, how it ended and what it
    kept. One of ``STATES``; an ended run's ``outcome`` is one of ``OUTCOMES``."""
    state: str = "idle"
    word: str = ""
    name: str = ""
    run: str | None = None
    place: tuple[str, str | None] | None = None
    loading: bool = False
    state_word: str = ""
    stage: str | None = None
    stage_words: str = ""
    needs_you: str | None = None
    outcome: str | None = None
    started: datetime | None = None
    ended: datetime | None = None
    last_heard: datetime | None = None
    kept: str = ""
    acknowledged: bool = False
    continue_from: int | None = None
    kept_fit: object | None = None
    stop_requested: bool = False
    closing: bool = False
    kinds: tuple[str, ...] = ()
    records: tuple[str, ...] = ()

    @property
    def live(self) -> bool:
        """The run is starting, running, stopping or waiting to try again."""
        return self.state in ("starting", "running", "stopping", "retrying")

    @property
    def unacknowledged_end(self) -> bool:
        """The run has ended, or was not seen to end, and nobody has seen how yet."""
        return self.state in ("ended", "not_seen") and not self.acknowledged


IDLE = RunStatus()


@dataclass(frozen=True)
class StageWord:
    """A stage on the stage line: its name, its marker ("done", "now", "later", "none", "half") and its words."""
    name: str
    marker: str
    word: str


@dataclass(frozen=True)
class AxisTimes:
    """The time axis of the stage running: its start, the present, the earliest and the latest end (the latest None
    when no end is known, the earliest then the moment it cannot end before, or None), the words said when no end
    is known, and the line under the axis that names what follows."""
    start: datetime
    now: datetime
    end_from: datetime | None
    end_by: datetime | None
    unknown: str
    follows: str


@dataclass(frozen=True)
class HealthRow:
    """One line of the training's health: what is measured, its number now, and its state."""
    measure: str
    now: str
    state: str


@dataclass(frozen=True)
class Message:
    """One message: its kind (a marker: "warn", "err", "info"), its sentence, when it was last said, and how many
    lines it stands for."""
    kind: str
    sentence: str
    when: datetime
    count: int


@dataclass(frozen=True)
class EpochRecord:
    """The record of one epoch of the fit, and when the window read it."""
    epoch: int
    training_loss: float
    validation_loss: float
    seconds: float
    since_best: int
    at: datetime


def driving_row(rows) -> object | None:
    """The progress row that measures the run: the largest total among the rows that can measure anything, the
    solver's own row left out (its total is in the tens of thousands and would win every time), a tie going to the
    deeper row. None when no row can measure."""
    return max((row for row in rows if row.informative and not row.is_solver),
               key=lambda row: (row.total, row.row), default=None)


def fit_counter(rows) -> int | None:
    """The epochs the training library's own counter row says it has trained, or None when no row is that
    counter. Its counter row is the fit's only sign when the library writes no record per epoch."""
    for row in rows:
        found = _COUNTER.fullmatch(row.desc.strip())
        if found is not None:
            return int(found.group(1))
    return None


def moved(before: int, after: int) -> bool:
    """Whether the solver's step count moved between two readings: a sign of life, printing or not."""
    return after > before


class RunModel:
    """One run, as its lines and rows tell it. Fed on the window's thread; every moment from ``clock``."""

    def __init__(self, spec: RunSpec, *, clock: Callable[[], datetime],
                 estimate: Callable[[str, dict], tuple[float, str] | None] | None = None,
                 expected: tuple[int, str] | None = None) -> None:
        self.spec = spec
        self._clock = clock
        self._estimate = estimate
        self._estimates: dict = {}
        if expected is None and spec.expected_epochs:
            expected = (spec.expected_epochs,
                        f"about {spec.expected_epochs} epochs: the number given on the Training page")
        self._expected = expected
        self.started = clock()
        self.kept_fit = None
        self._state, self._outcome = "starting", None
        self._heard = self.started
        self._steps: int | None = None
        self._stages = list(spec.stages)
        self._began: dict[int, datetime] = {}
        self._done: set[int] = set()
        self._left_out: set[int] = set()
        self._current: int | None = None
        self._rows: tuple = ()
        self._samples: dict = {}            # a counted stage's index, or _OWN: [(when, done)]
        self._totals: dict = {}             # the same keys: the total counted to
        self._unit = ""
        self._budget: dict | None = None
        self._folder: str | None = None
        self._every: int | None = None
        self._resumed_at: int | None = None
        self._rotation: str | None = None
        self._not_needed = False
        self._records: dict[int, EpochRecord] = {}
        self._counter: list[tuple[datetime, int]] = []
        self._resumed_fit: tuple[int, int | None] | None = None     # the pass resumed after, its count since the best
        self._fit_ended_after: int | None = None
        self._failed: tuple[int, int] | None = None
        self._earlier: tuple[tuple[int, int], ...] | None = None
        self._unusable: tuple[int, int] | None = None
        self._unusable_counted = False
        self._chi_seen = False
        self._peak: float | None = None
        self._card: float | None = None
        self._memory_trouble = False
        self._saved: int | None = None
        self._save_failed = False
        self._saving_shown = False
        self._not_saved = False
        self._kept: list[tuple[str, Message]] = []
        self._silence: Message | None = None
        self._silence_needs_you = False

    # ── feeds ──
    def set_earlier(self, earlier: tuple[tuple[int, int], ...]) -> None:
        """The failed and the simulated counts of every earlier run of this run's simulation set, oldest first:
        the failed simulations are then counted over all of them."""
        self._earlier = tuple((int(bad), int(rows)) for bad, rows in earlier)

    def feed(self, lines: Iterable[tuple[str, str]]) -> None:
        """The run's lines, each ``(text, level)``, as its streams carry them."""
        for text, level in lines:
            self._hear()
            reading = readings.read(text, level)
            if reading is not None:
                self._take(reading)
            self._say(text, level, reading)

    def rows(self, rows: Sequence) -> None:
        """The run's live progress rows, each read by its attributes (``desc``, ``stats``, ``total``, ``row``,
        ``informative``, ``is_solver``)."""
        self._rows = tuple(rows)
        self._hear()
        descs = {row.desc.strip(): row for row in self._rows}
        counted = self._counted_by(lambda stage: stage.counts in descs)
        if counted is not None:
            self._count_row(counted, descs[self._stages[counted].counts])
        elif not self._stages and driving_row(self._rows) is not None:
            self._count_row(_OWN, driving_row(self._rows))
        epochs = fit_counter(self._rows)
        if epochs is not None:
            fit = self._kind_index("fit")
            if fit is not None:
                self._begin(fit)
            if not self._counter or self._counter[-1][1] != epochs:
                self._counter.append((self._clock(), epochs))
        for row in self._rows:
            reading = readings.read(row.desc, "info")
            if reading is not None and reading.name == "sbi_converged":
                self._take(reading)

    def solver(self, steps: int) -> None:
        """The solver's step count now; a count that moved is a sign of life."""
        if self._steps is not None and moved(self._steps, steps):
            self._hear()
        self._steps = steps

    def tick(self) -> None:
        """Look at the time: a silence is said, and grows, from here."""
        self._silence_needs_you = False
        if self._state not in ("starting", "running", "stopping"):
            self._silence = None
            return
        now = self._clock()
        quiet = (now - self._heard).total_seconds()
        if quiet < SILENCE_ATTENTION_S:
            self._silence = None
            return
        words = _quiet_words(quiet)
        if self._in_fit():
            epoch = self._epoch_seconds()
            if epoch is not None and quiet >= 2 * epoch:
                kind, sentence = "warn", f"No output for {words}, twice as long as the last epoch took."
                self._silence_needs_you = True
            else:
                kind, sentence = "info", f"No output for {words}. This is normal during an epoch."
        else:
            kind, sentence = "warn", f"No output for {words}."
            self._silence_needs_you = quiet >= SILENCE_NEEDS_YOU_S
        when = self._silence.when if self._silence is not None and self._silence.kind == kind else now
        self._silence = Message(kind, sentence, when, 1)

    def set_count(self, done: int, total: int, unit: str) -> None:
        """A run that prints no counted row says how far it is itself: ``done`` of ``total`` ``unit``."""
        self._hear()
        self._unit = unit
        self._add_sample(_OWN, int(done), int(total))

    def set_stage(self, name: str) -> None:
        """A run that says its stages itself begins the one of that name (a name its spec does not hold is added
        after the others)."""
        self._hear()
        index = next((i for i, stage in enumerate(self._stages) if stage.name == name), None)
        if index is None:
            self._stages.append(Stage(name))
            index = len(self._stages) - 1
        self._begin(index)

    def stopping(self, on: bool) -> None:
        """A stop was asked for (True), or withdrawn (False)."""
        if self._state in ("starting", "running", "stopping"):
            self._state = "stopping" if on else "running"

    def ended(self, outcome: str, *, kept_fit=None) -> None:
        """The run has ended: one of ``OUTCOMES``, with the kept training's note when the run kept one. A finished
        run's stages are all done but those passed over; after any end nothing begins or moves a stage."""
        if outcome not in OUTCOMES:
            raise ValueError(f"{outcome!r} is not how a run ends: {OUTCOMES}")
        self._state, self._outcome, self.kept_fit = "ended", outcome, kept_fit
        self._silence, self._silence_needs_you = None, False
        if outcome == "finished":
            latest = max(self._began, default=-1)
            self._done.update(index for index in range(len(self._stages)) if index in self._began or index > latest)

    # ── read-outs ──
    @property
    def state(self) -> str:
        """"starting" until the run's first line or row, then "running", "stopping" while a stop is asked for, and
        "ended"."""
        return self._state

    @property
    def outcome(self) -> str | None:
        """How the run ended, one of ``OUTCOMES``, or None while it runs."""
        return self._outcome

    @property
    def stage(self) -> str | None:
        """The name of the stage the run is in, or None before its first."""
        return None if self._current is None else self._stages[self._current].name

    def stage_kind(self) -> str | None:
        """What the stage the run is in is: "rotation", "simulation", "fit" or "saving" (the saving of the
        posterior), read off what begins and counts it; None before the run's first stage and for a stage of another
        kind of run."""
        return self._kind_of(self._current)

    def now(self) -> datetime:
        """The moment now, on the run's clock."""
        return self._clock()

    def stop_bound(self) -> str | None:
        """How long a stop asked for during the fit may take to be taken, in words ("63 minutes"): the fit's pace, as
        the stage line says it; None while no pace is known."""
        pace = self._epoch_pace()
        return None if pace is None else _pace_words(pace)

    def last_epoch_ends(self) -> datetime | None:
        """When the fit ends by itself if validation does not improve, while the epoch in progress may be its last:
        the newest epoch's count since the best is one short of the patience. None otherwise, and while no pace of
        the fit is known."""
        patience = self.spec.patience
        if not self._in_fit() or not self._records or not patience:
            return None
        newest, pace = self._records[max(self._records)], self._epoch_pace()
        if newest.since_best != patience - 1 or pace is None:
            return None
        return newest.at + timedelta(seconds=pace)

    def sentence(self) -> str:
        """The one sentence that says where the run stands."""
        who = f"{self.spec.word} {self.spec.name}".strip()
        if self._state == "ended":
            return _ENDED_SENTENCES[self._outcome].format(who=who)
        kind = self._kind_of(self._current)
        if self._state == "stopping":
            if kind == "simulation":
                return "Stopping: the finished batches are being saved."
            if kind == "fit":
                bound = self.stop_bound()
                return "Stopping: takes effect when this epoch ends" + (f", up to about {bound}." if bound else ".")
            return "Stopping: takes effect at the run's next line of output."
        now = self._clock()
        if kind == "rotation":
            if self._rotation == "computing":
                return "The rotation is being computed before any simulation."
            return f"{who} is starting."
        if kind == "simulation":
            done, total = self._numbers(self._current)
            pace = self._pace(self._current)
            if total is None:
                return f"{who} is starting."
            if pace is None:
                return f"Simulating: batch {number_text(done)} of {number_text(total)}."
            end = now + timedelta(seconds=(total - done) * pace)
            return (f"Simulating: batch {number_text(done)} of {number_text(total)}, {pace:.1f} s per batch; the "
                    f"simulations should be done by {date_text(end, now)}.")
        if kind == "fit":
            return self._fit_sentence(now)
        if kind == "saving":
            return (f"The fit ended after {number_text(self._fit_ended_after or self._last_epoch())} epochs. The "
                    f"posterior is being saved.")
        if self._current is not None:
            return f"{who} is running: {self._stages[self._current].name}."
        if _OWN in self._samples:
            done, total = self._numbers(_OWN)
            unit = f" {self._unit}" if self._unit else ""
            return f"{who} is running: {number_text(done)} of {number_text(total)}{unit}."
        return f"{who} is starting." if self._stages or self._state == "starting" else f"{who} is running."

    def stage_line(self) -> tuple[StageWord, ...]:
        """The run's stages with their markers and words, those left out not drawn."""
        out = []
        latest = max(self._began, default=-1)
        for index, stage in enumerate(self._stages):
            if index in self._left_out:
                continue
            kind = self._kind_of(index)
            passed = index not in self._began and latest > index
            if passed or (kind == "rotation" and self.spec.facts.get("rotated") is False):
                out.append(StageWord(stage.name, "none", "Off"))
            elif index not in self._began and index not in self._done:
                out.append(StageWord(stage.name, "later", "To come"))
            elif index in self._done or index < latest:
                out.append(StageWord(stage.name, "done", self._done_word(index, kind)))
            elif self._state == "ended":
                out.append(StageWord(stage.name, "half", _ENDED_WORDS.get(self._outcome, "Done")))
            else:
                pace = self._epoch_pace() if kind == "fit" else None
                out.append(StageWord(stage.name, "now",
                                     f"Now, about {_pace_words(pace)} per epoch" if pace is not None else "Now"))
        return tuple(out)

    def axis(self) -> AxisTimes | None:
        """The time axis of the stage running, or of the count of a run with no stages, from the run's start; None
        when there is none (the stage counts nothing and is not the fit, it is done, or the run has ended)."""
        if self._state == "ended":
            return None
        now = self._clock()
        if self._current is None:
            if self._stages or _OWN not in self._samples:
                return None
            return self._counted_axis(_OWN, self.started, now, "")
        if self._current in self._done:
            return None
        kind = self._kind_of(self._current)
        if kind == "fit":
            return self._fit_axis(now)
        if kind != "simulation" and self._stages[self._current].counts is None:
            return None
        return self._counted_axis(self._current, self._began[self._current], now,
                                  self._follows() if kind == "simulation" else "")

    def health(self) -> tuple[HealthRow, ...]:
        """The training's health, for a run that simulates training rows; nothing for any other run."""
        if not any(stage.counts == SIMULATION_ROW for stage in self._stages):
            return ()
        out = [self._failed_row()]
        if self.spec.facts.get("observation") == "chi" or self._chi_seen:
            if self._unusable is not None:
                masked, total = self._unusable
                out.append(HealthRow("Unusable driven recordings",
                                     f"{number_text(round(100.0 * masked / total, 1))} %", "Not compared"))
            elif self._unusable_counted:
                out.append(HealthRow("Unusable driven recordings", "None counted", "Not compared"))
            else:
                out.append(HealthRow("Unusable driven recordings", "Counted when simulation ends", ""))
        out.append(self._memory_row())
        if self._saving_shown and not self._not_saved:
            if not self._save_failed:
                out.append(HealthRow("Training saved", f"after epoch {number_text(self._saved)}", "Fine"))
            elif self._saved is not None:
                out.append(HealthRow("Training saved", f"epoch {number_text(self._saved)}; the last save failed",
                                     "Look closer"))
            else:
                out.append(HealthRow("Training saved", "nothing kept yet; the last save failed", "Look closer"))
        return tuple(out)

    def messages(self) -> tuple[Message, ...]:
        """The messages, newest first: a silence while it lasts, then the newest of the rest."""
        return ((self._silence,) if self._silence is not None else ()) + tuple(m for _, m in self._kept)

    def needs_you(self) -> str | None:
        """Why the run needs someone now, or None."""
        return self._silence.sentence if self._silence_needs_you and self._silence is not None else None

    def epochs(self) -> tuple[EpochRecord, ...]:
        """The records of the fit's epochs read so far, in order."""
        return tuple(self._records[epoch] for epoch in sorted(self._records))

    def best_epoch(self) -> int | None:
        """The fit's best epoch so far, as its newest record counts it (a resumed fit's earlier epochs included)."""
        if not self._records:
            return None
        newest = self._records[max(self._records)]
        return newest.epoch - newest.since_best

    def batches_done(self) -> int | None:
        """The batches of the simulation set done so far, or None before the simulation."""
        simulation = self._kind_index("simulation")
        samples = self._samples.get(simulation) if simulation is not None else None
        return samples[-1][1] if samples else self._resumed_at

    def simulation_folder(self) -> str | None:
        """The simulation set's folder, as the run names it, or None."""
        return self._folder

    def began_after_first(self) -> bool:
        """Whether a stage after the run's first has begun."""
        return any(index > 0 for index in self._began)

    def fit_not_saved(self) -> bool:
        """Whether the run said its fit is not saved as it goes."""
        return self._not_saved

    def saves_passed(self) -> int:
        """The saves of the simulation set this run has passed: how many multiples of the set's cadence the
        batches done have reached since the run began. For display only: the engine may put a save off."""
        every = self._every or self.spec.facts.get("cadence")
        done = self.batches_done()
        if not every or done is None:
            return 0
        return done // every - (self._resumed_at or 0) // every

    def budget(self) -> dict | None:
        """The run's budget as its line gave it: ``"batches"``, ``"rows"`` (simulations per batch) and
        ``"simulations"`` in all; None before the line."""
        return None if self._budget is None else dict(self._budget)

    def failed(self) -> tuple[int, int] | None:
        """The failed simulations and the simulations made, as this run has counted them so far; None before its
        first count."""
        return self._failed

    def seconds_per_batch(self) -> float | None:
        """The simulation's own pace, in seconds per batch, once its first ``TIMED_BATCHES`` are timed; None before,
        and for a run with no simulation. Never the history's estimate."""
        simulation = self._kind_index("simulation")
        samples = self._samples.get(simulation) if simulation is not None else None
        if not samples or samples[-1][1] - samples[0][1] < self._timed(simulation):
            return None
        return self._pace(simulation)

    # ── reading a line ──
    def _take(self, reading: readings.Reading) -> None:
        name, values = reading.name, reading.values
        for index, stage in enumerate(self._stages):
            if name in stage.begins and index not in self._began and (self._current is None
                                                                       or index > self._current):
                self._begin(index)
                break
        counted = self._counted_by(lambda stage: stage.counts == name) if {"i", "n"} <= values.keys() else None
        if counted is not None:
            self._begin(counted)
            self._add_sample(counted, values["i"], values["n"])
        if name in LEAVES_OUT and self._current is not None and self._state != "ended":
            self._left_out.add(self._current)
        if name == "budget":
            self._budget = values
        elif name == "rotation_begins":
            self._rotation = "computing"
        elif name in ("rotation_done", "rotation_reused", "rotation_parent"):
            self._rotation = {"rotation_done": "done", "rotation_reused": "reused",
                              "rotation_parent": "parent"}[name]
            self._close(self._kind_index("rotation"), begin=True)
            if name == "rotation_reused" and values["complete"]:
                self._not_needed = True
        elif name == "simulation_writing":
            self._folder, self._every = values["dir"], values["every"]
        elif name == "simulation_resuming":
            self._folder, self._resumed_at = values["dir"], values["k"]
            self._not_needed = self._not_needed or values["k"] >= values["n"]
        elif name == "simulation_complete":
            self._close(self._kind_index("simulation"))
        elif name == "fit_epoch":
            self._record(values)
        elif name == "sbi_converged":
            self._fit_ended_after = values["n"]
            self._end_fit()
        elif name == "patho_batch":
            self._failed = (values["nonfinite"] + values["constant"] + values["overflow"], values["rows"])
        elif name == "patho_total":
            self._failed = (values["bad"], values["rows"])
        elif name in ("chi_batch", "chi_total", "chi_none"):
            self._chi_seen = True
            if name == "chi_total":
                self._unusable, self._unusable_counted = (values["masked"], values["total"]), True
            elif name == "chi_none":
                self._unusable, self._unusable_counted = None, True
        elif name == "mem":
            self._peak = max(self._peak or 0.0, values["peak"])
            self._card = values["card"]
        elif name in ("oom_split_simulation", "oom_split_rows", "batch_failed"):
            self._memory_trouble = True
        elif name in ("fit_saving_off", "fit_not_kept"):
            self._not_saved = True
        elif name in ("fit_saved", "fit_resuming"):
            self._saved, self._save_failed, self._saving_shown = values["passes"], False, True
            if name == "fit_resuming":
                since = _SINCE.search(values["progress"])
                self._resumed_fit = (values["passes"], None if since is None else int(since.group(1)))
        elif name in ("fit_stopped_kept", "fit_stopped_not_saved_here"):
            self._saved, self._saving_shown = values["passes"], True
        elif name == "fit_save_failed":
            self._saved, self._save_failed, self._saving_shown = values["passes"], True, True
        elif name == "fit_save_failed_nothing_kept":
            self._save_failed, self._saving_shown = True, True

    def _record(self, values: dict) -> None:
        record = EpochRecord(values["epoch"], values["training_loss"], values["validation_loss"], values["seconds"],
                             values["since"], self._clock())
        self._records[record.epoch] = record
        fit = self._kind_index("fit")
        if fit is not None:
            self._begin(fit)
        patience, limit = self.spec.patience, self.spec.epoch_limit
        if ((patience is not None and record.since_best >= patience)
                or (limit is not None and record.epoch >= limit + 1)):
            self._fit_ended_after = record.epoch
            self._end_fit()

    def _end_fit(self) -> None:
        for index, stage in enumerate(self._stages):
            if (FIT_ENDED in stage.begins or "sbi_converged" in stage.begins) and index not in self._began:
                self._begin(index)
                return

    def _say(self, text: str, level: str, reading) -> None:
        """A message, when the line is one: a warning or an error, in the window's own words where its row has
        them and from the engine as written where it has none, or a line of information its row makes one."""
        row = None if reading is None else next(r for r in readings.ROWS if r.name == reading.name)
        if level in ("warning", "error"):
            kind = "err" if level == "error" else "warn"
        elif level == "info" and row is not None and row.message == "information":
            kind = "info"
        else:
            return
        if reading is not None and reading.name == "chi_batch":
            key = "chi_batch"                        # one message, its count in its sentence: never summed
            count = 1 + next((m.count for k, m in self._kept if k == key), 0)
            sentence = readings.sentence(reading, count=count)
        else:
            sentence = None if row is None else readings.sentence(reading)
            if sentence is None:
                line = text.strip()
                sentence = f"From the engine: {line}" + ("" if line.endswith((".", "!", "?", "…")) else ".")
            key = _DIGITS.sub("#", sentence)        # differing only in their numbers: one message, counted
            count = 1 + next((m.count for k, m in self._kept if k == key), 0)
        self._kept = [(key, Message(kind, sentence, self._clock(), count))] + [(k, m) for k, m in self._kept
                                                                                 if k != key]
        del self._kept[MESSAGES_KEPT:]

    # ── the stages ──
    def _begin(self, index: int) -> None:
        if index in self._began or self._state == "ended":
            return
        for earlier in self._began:
            if earlier < index:
                self._done.add(earlier)
        self._began[index] = self._clock()
        if self._current is None or index > self._current:
            self._current = index
        if self._state == "starting":
            self._state = "running"

    def _close(self, index: int | None, *, begin: bool = False) -> None:
        """A stage is done, begun first when ``begin``; once the run has ended nothing moves."""
        if index is None or self._state == "ended":
            return
        if begin:
            self._begin(index)
        self._done.add(index)

    def _hear(self) -> None:
        self._heard = self._clock()
        self._silence, self._silence_needs_you = None, False
        if self._state == "starting":
            self._state = "running"
        if (self._state in ("running", "stopping", "retrying") and not self._began and self._stages
                and not self._stages[0].begins):
            self._begin(0)                  # with the run's first output, a stop already asked for or not

    def _kind_of(self, index) -> str | None:
        """What a stage is, read off what begins and counts it: the rotation, the simulation, the fit, the saving
        of the posterior, or None for a stage of another run."""
        if index is None or index == _OWN:
            return None
        stage = self._stages[index]
        if "rotation_begins" in stage.begins:
            return "rotation"
        if stage.counts == SIMULATION_ROW:
            return "simulation"
        if stage.counts == "fit_epoch":
            return "fit"
        if FIT_ENDED in stage.begins or "sbi_converged" in stage.begins:
            return "saving"
        return None

    def _kind_index(self, kind: str) -> int | None:
        return next((i for i in range(len(self._stages)) if self._kind_of(i) == kind), None)

    def _counted_by(self, counts) -> int | None:
        """The stage a count belongs to: the stage running, when it is counted so; else the first stage after it,
        not yet begun, that is. Two stages counted alike (the two sweeps of a study) are counted in turn."""
        if self._current is not None and counts(self._stages[self._current]):
            return self._current
        return next((i for i, stage in enumerate(self._stages) if counts(stage) and i not in self._began
                     and (self._current is None or i > self._current)), None)

    def _done_word(self, index: int, kind: str | None) -> str:
        if kind == "rotation":
            return _ROTATION_WORDS.get(self._rotation, "Done")
        if kind == "simulation" and self._not_needed:
            return "Not needed: the saved simulations are used"
        if kind == "simulation" and self._budget is not None:
            return f"Done, {number_text(self._budget['simulations'])} simulations"
        return "Done"

    def _in_fit(self) -> bool:
        return self._kind_of(self._current) == "fit" and self._current not in self._done

    # ── counts, paces and ends ──
    def _count_row(self, key, row) -> None:
        found = _DONE_OF.match(row.stats or "")
        if found is None:
            return
        if key != _OWN:
            self._begin(key)
        self._add_sample(key, int(found.group(1)), int(found.group(2)))

    def _add_sample(self, key, done: int, total: int) -> None:
        self._totals[key] = total
        samples = self._samples.setdefault(key, [])
        if not samples or samples[-1][1] != done:
            samples.append((self._clock(), done))

    def _numbers(self, key) -> tuple[int, int | None]:
        """``(done, total)`` of a counted stage, or of the run's own count: the simulation's from its row, its
        resume or its budget before the row's first frame."""
        samples = self._samples.get(key) or []
        total = self._totals.get(key)
        if self._kind_of(key) == "simulation":
            total = total or (self._budget or {}).get("batches")
            return (samples[-1][1] if samples else self._resumed_at or 0), total
        return (samples[-1][1] if samples else 0), total

    def _pace(self, key) -> float | None:
        """Seconds per count over the newest ``PACE_BATCHES`` once enough are timed (``_timed``); before that, for
        the simulation, the history's estimate."""
        samples = self._samples.get(key) or []
        if samples and samples[-1][1] - samples[0][1] >= self._timed(key):
            last_at, last = samples[-1]
            first_at, first = next((at, done) for at, done in samples if done >= last - PACE_BATCHES)
            if last > first:
                return (last_at - first_at).total_seconds() / (last - first)
        if self._kind_of(key) == "simulation":
            estimated = self._estimated("batch")
            return None if estimated is None else estimated[0]
        return None

    def _timed(self, key) -> int:
        """The steps a count times before its own pace is taken: the simulation's ``TIMED_BATCHES``; any other
        count's total divided by ``SHORT_COUNT_PARTS`` when that is fewer, ``SHORT_COUNT_LEAST`` at the least."""
        total = self._totals.get(key)
        if self._kind_of(key) == "simulation" or not total:
            return TIMED_BATCHES
        return min(TIMED_BATCHES, max(SHORT_COUNT_LEAST, total // SHORT_COUNT_PARTS))

    def _counted_axis(self, key, start: datetime, now: datetime, follows: str) -> AxisTimes:
        """The axis of a count by the simulation's rule: firm, now plus what is left at its own pace, or no end while
        too few of its steps are timed."""
        done, total = self._numbers(key)
        pace = self._pace(key)
        if total is None or pace is None:
            unknown = ("No end time yet: the first batches are being timed." if self._kind_of(key) == "simulation"
                       else "No end time yet: the first steps are being timed.")
            return AxisTimes(start, now, None, None, unknown, follows)
        end = now + timedelta(seconds=(total - done) * pace)
        return AxisTimes(start, now, end, end, "", follows)

    def _size(self) -> dict:
        budget = self._budget or {}
        return {"per_batch": budget.get("rows"), "simulations": budget.get("simulations"),
                "network": self.spec.facts.get("network")}

    def _estimated(self, per: str) -> tuple[float, str] | None:
        """The history's pace of a batch or an epoch at this run's size, asked once for each size."""
        size = self._size()
        known = (size["per_batch"],) if per == "batch" else (size["simulations"], size["network"])
        if self._estimate is None or any(value is None for value in known):
            return None
        key = (per, repr(sorted(size.items())))
        if key not in self._estimates:
            self._estimates[key] = self._estimate(per, size)
        return self._estimates[key]

    def _last_epoch(self) -> int:
        if self._records:
            return max(self._records)
        if self._counter:
            return self._counter[-1][1]
        return self._resumed_fit[0] if self._resumed_fit is not None else 0

    def _epoch_seconds(self) -> float | None:
        """The last epoch's time, or the history's pace before the first."""
        if self._records:
            return self._records[max(self._records)].seconds
        if len(self._counter) >= 2:
            return (self._counter[-1][0] - self._counter[-2][0]).total_seconds()
        estimated = self._estimated("epoch")
        return None if estimated is None else estimated[0]

    def _epoch_pace(self) -> float | None:
        """The median time of the newest ``FIT_PACE_RECORDS`` epochs, or the history's pace before the first."""
        if self._records:
            newest = [self._records[e].seconds for e in sorted(self._records)[-FIT_PACE_RECORDS:]]
            return float(statistics.median(newest))
        if len(self._counter) >= 2:
            gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(self._counter, self._counter[1:])]
            return float(statistics.median(gaps[-FIT_PACE_RECORDS:]))
        estimated = self._estimated("epoch")
        return None if estimated is None else estimated[0]

    def _fit_numbers(self):
        """``(n, since, newest, start)``: the epochs finished, the epochs since the best (None before the first, or
        when no loss of a resumed fit was finite), when the newest epoch was read, and when the fit began in this
        process: the first record's time less its own, else when the stage began. Before its first record here, a
        resumed fit counts from the pass it resumed after."""
        began = self._began[self._kind_index("fit")]
        if self._records:
            first, newest = self._records[min(self._records)], self._records[max(self._records)]
            return newest.epoch, newest.since_best, newest.at, first.at - timedelta(seconds=first.seconds)
        if self._counter:
            return self._counter[-1][1], 0, self._counter[-1][0], began
        if self._resumed_fit is not None:
            return *self._resumed_fit, began, began
        return 0, None, began, began

    def _fit_ends(self):
        """``(cannot, earliest, latest)``: when the fit cannot end before, and the earliest and the latest end of the
        expected range (None without the expected epochs; once the fit has run past the range, both the moment it
        was due, so that the axis is late). All None when no pace is known."""
        pace = self._epoch_pace()
        if pace is None:
            return None, None, None
        n, since, newest, start = self._fit_numbers()
        patience, limit = self.spec.patience or 0, self.spec.epoch_limit

        def at(epochs):
            return newest + timedelta(seconds=epochs * pace)

        cap = None if limit is None else limit + 1 - n                 # the epochs left before the limit stops it
        left = patience + 1 if since is None else patience - since    # the first epoch only sets the best
        if cap is not None:
            left = min(left, cap)
        left = max(left, 0)
        if self._expected is None:
            return at(left), None, None
        expected = self._expected[0]
        if n > expected + patience:                                    # due when the range ran out
            closed = self._records.get(expected + patience)            # not read here: no earlier than the start
            due = closed.at if closed is not None else max(at(expected + patience - n), start)
            return at(left), due, due
        earliest, latest = max(left, expected - patience - n), expected + patience - n
        if cap is not None:
            earliest, latest = min(earliest, cap), min(latest, cap)
        return at(left), at(earliest), at(max(latest, earliest))

    def _fit_axis(self, now) -> AxisTimes:
        start = self._fit_numbers()[3]
        follows = ""
        if self._expected is not None:
            words = self._expected[1]
            follows = words[:1].upper() + words[1:] + "."
        cannot, earliest, latest = self._fit_ends()
        if cannot is None:
            return AxisTimes(start, now, None, None, "No end time yet: the first epoch has not finished.", follows)
        if latest is None:
            return AxisTimes(start, now, cannot, None,
                             f"Cannot end before {date_text(cannot, now)}. No expected number of epochs was given, "
                             f"and there is no earlier training of this kind.", follows)
        return AxisTimes(start, now, earliest, latest, "", follows)

    def _fit_sentence(self, now) -> str:
        n = self._last_epoch()
        if n == 0:
            said = "The simulations are done. The fit has started; its first epoch has not finished."
            estimated = self._estimated("epoch")
            if estimated is not None:
                said += f" At this size an epoch last took about {_pace_words(estimated[0])}."
            return said
        cannot = self._fit_ends()[0]
        if self._expected is not None and n > self._expected[0] and cannot is not None:     # past the estimate
            return (f"Epoch {number_text(n)}. The estimate was about {number_text(self._expected[0])}. The fit ends "
                    f"by itself {number_text(self.spec.patience or 0)} epochs after its last improvement: not "
                    f"before {date_text(cannot, now)}.")
        of = f" of about {number_text(self._expected[0])}" if self._expected is not None else ""
        return f"Now: {date_text(now, now)}, running {_running_words(now - self.started)}, epoch {number_text(n)}{of}."

    def _follows(self) -> str:
        """While the simulation runs: how long the fit after it should take, from the history's pace of an epoch
        and the expected epochs, or nothing."""
        estimated = self._estimated("epoch")
        if estimated is None or self._expected is None:
            return ""
        expected, patience, limit = self._expected[0], self.spec.patience or 0, self.spec.epoch_limit
        least, most = max(expected - patience, patience + 1), expected + patience
        if limit is not None:
            least, most = min(least, limit + 1), min(most, limit + 1)
        return f"Then the fit: about {_span_words(least * estimated[0], most * estimated[0])}."

    # ── health ──
    def _failed_row(self) -> HealthRow:
        bad, rows = self._failed or (0, 0)
        over = ""
        if self._earlier is not None:
            bad += sum(b for b, _ in self._earlier)
            rows += sum(r for _, r in self._earlier)
            over = f" over {len(self._earlier) + 1} runs"
        elif self._resumed_at:
            over = f" since batch {number_text(self._resumed_at)}"
        if not rows:
            return HealthRow("Failed simulations", "None so far", "Fine")
        share = bad / rows
        return HealthRow("Failed simulations", f"{number_text(round(100.0 * share, 2))} %{over}",
                         "Fine" if share <= FAILED_FINE else "Look closer")

    def _memory_row(self) -> HealthRow:
        state = "Look closer" if self._memory_trouble else ""
        if self._peak is None:
            return HealthRow("Graphics-card memory", "Not measured yet", state)
        limit = self.spec.facts.get("memory_limit")
        said = f"{_gib(self._peak)} of {_gib(self._card)} GiB" + (f", limit {_gib(limit)} GiB" if limit else "")
        fit = self._kind_index("fit")
        if fit is not None and fit in self._began:
            said += ", during simulation"
        return HealthRow("Graphics-card memory", said, state or "Fine")


def _gib(value: float) -> str:
    """A size in GiB to one decimal, a whole one written whole: "10.4", "16"."""
    rounded = round(float(value), 1)
    return number_text(int(rounded)) if rounded == int(rounded) else number_text(rounded)


def _pace_words(seconds: float) -> str:
    """A pace's time: whole seconds under a minute, else whole minutes ("63 minutes")."""
    if seconds < 59.5:
        count, unit = max(1, round(seconds)), "second"
    else:
        count, unit = round(seconds / 60), "minute"
    return f"{count} {unit}{'' if count == 1 else 's'}"


def _running_words(elapsed: timedelta) -> str:
    """How long the run has run: whole hours from one hour ("65 hours"), as a shorter length is written below."""
    seconds = elapsed.total_seconds()
    if seconds >= 3600:
        hours = math.floor(seconds / 3600)
        return f"{hours} hour{'' if hours == 1 else 's'}"
    return duration_text(seconds)


def _quiet_words(seconds: float) -> str:
    """A silence's length: whole minutes under an hour, one at the least and 59 at the most; as a length is written
    above."""
    if seconds < 3600:
        minutes = min(59, max(1, round(seconds / 60)))
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    return duration_text(seconds)


def _span_words(least: float, most: float) -> str:
    """A range of lengths in the unit of the shorter: "5 to 7 days", "3 to 4 hours", "40 to 55 minutes"; one number
    when both come to it."""
    unit, size = next((unit, size) for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60))
                      if least >= size or unit == "minute")
    low, high = max(1, round(least / size)), max(1, round(most / size))
    plural = "" if high == 1 else "s"
    return f"{low} {unit}{plural}" if low == high else f"{low} to {high} {unit}{plural}"
