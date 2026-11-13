"""The scratch store of the two checks that keep nothing but their result, the timing of a short fit and the card
test: a store of their own, in a folder of the computer's temporary files made for the one run, handed to the stage
as its store and removed however the run ends. The process's store is never swapped for it, since every page reads
that one, and nothing either check makes reaches the records in use.

What a check's run reads of its own records it reads as they are said, by a handler on the engine's logger that only
notes; what it hands back is its result as names and numbers, which its run's history keeps. Both calls run on the
worker's thread.
"""
import gc
import logging
import shutil
import statistics
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.artifacts import ArtifactStore

from .check_readings import CARD_STAGE_WORDS  # noqa: F401 -- the card test's stages in the words its page says them
from .run import readings

log = logging.getLogger(__name__)

#: The engine's records that end the generation of a training's rows, by the window's table of its lines: the total of
#: failed simulations, and the total of unusable driven recordings in either of its forms.
GENERATED = ("patho_total", "chi_total", "chi_none")
#: The records that say a batch was run in smaller pieces for lack of the graphics card's memory.
SPLIT = ("oom_split_simulation", "oom_split_rows")

#: The name a scratch folder begins with, among the computer's temporary files.
SCRATCH_PREFIX = "prism_check_"
#: How many times in all, and how many seconds apart, a scratch folder's removal is tried before it is given up and
#: said: Windows may hold a file of it open for a moment after the run has let it go (a scanner, an indexer).
RMTREE_TRIES = 5
RMTREE_PAUSE_S = 0.2
#: What the run's text says, once, of a scratch folder that could not be removed.
LEFT_TEXT = "The scratch folder %s of a check that keeps nothing could not be removed (%s); it can be deleted by hand."


@contextmanager
def scratch_store(*, prefix: str = SCRATCH_PREFIX, pause=time.sleep):
    """A store of its own in a new temporary folder whose name begins ``prefix``, for the block; the folder removed
    when the block ends, however it ends (finished, failed, stopped). The caller drops its references to what the
    store loaded before the block ends. A removal Windows refuses is tried again, ``RMTREE_TRIES`` times in all,
    ``pause`` (``RMTREE_PAUSE_S`` seconds) between two, what is no longer referenced collected first; a folder that
    still cannot go is said once, as a warning that names it, and the block's own end passes on unchanged."""
    folder = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield ArtifactStore(folder)
    finally:
        _remove(folder, pause)


def _remove(folder: Path, pause) -> None:
    """Remove ``folder`` and all it holds, as ``scratch_store`` says; never raises."""
    error = None
    for tried in range(1, RMTREE_TRIES + 1):
        try:
            shutil.rmtree(folder)
            return
        except FileNotFoundError:
            return
        except OSError as refused:
            error = refused
            if tried < RMTREE_TRIES:
                gc.collect()                 # a record let go but still held in a cycle may hold a file of it open
                pause(RMTREE_PAUSE_S)
    if folder.exists():
        log.warning(LEFT_TEXT, folder, f"{type(error).__name__}: {error}")


class _Noted(logging.Handler):
    """What a check that keeps nothing reads of its run's records as they are said, by the window's table of the
    engine's lines: when the first record that ends the generation of a training's rows was made (seconds since the
    epoch, the record's own moment); the seconds of each epoch's record; the first total of unusable driven recordings
    (its unusable and its total; none counted, nought of nought); the batches run in smaller pieces; and the
    simulations the budget names. It only notes, and never raises."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.generated: "float | None" = None
        self.took: list = []
        self.masked: "tuple[int, int] | None" = None
        self.split: set = set()
        self.simulations: "int | None" = None

    def emit(self, record) -> None:
        try:
            reading = readings.read(record.getMessage(), record.levelname.lower())
            if reading is None:
                return
            name, values = reading.name, reading.values
            if name in GENERATED and self.generated is None:
                self.generated = record.created
            if name == "fit_epoch":
                self.took.append(values["seconds"])
            elif name in ("chi_total", "chi_none") and self.masked is None:
                self.masked = (values["masked"], values["total"]) if name == "chi_total" else (0, 0)
            elif name in SPLIT:
                self.split.add((values.get("batches"), values.get("batch")))
            elif name == "budget" and self.simulations is None:
                self.simulations = values["simulations"]
        except Exception:                        # noqa: BLE001 -- what only notes never stops a run
            pass


@contextmanager
def _noting(handler: _Noted):
    """``handler`` on the engine's logger for the block, and off it however the block ends."""
    held = logging.getLogger("core")
    held.addHandler(handler)
    try:
        yield handler
    finally:
        held.removeHandler(handler)


def _moment(created) -> "float | None":
    """A record's moment as its manifest gives it, in seconds since the epoch; None when it gives none that reads."""
    try:
        return datetime.fromisoformat(created).timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _count(value) -> "int | None":
    """A count a record holds, or None for anything that is no whole number."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class ShortFit:
    """What a short fit hands back: its result, names and numbers, which its run's history keeps. It names no record:
    the record it made went with its scratch store."""
    check_result: dict


def short_fit(cfg, prior, *, num_runs: int, run_size_cap: int, epochs: int, hidden_features: int,
              num_transforms: int, press: str = "", fig_sink=None, store_factory=scratch_store) -> ShortFit:
    """One short fit of the network ``hidden_features`` by ``num_transforms`` on ``num_runs`` batches of at most
    ``run_size_cap`` simulations, for at most ``epochs`` epochs (the limit the stage is sent is one less, as it trains
    one more), on the prior ``prior`` handed as the object, in a scratch store: its simulations not kept, so no saved
    set is read or written and no training is kept as it goes. Its result: the network, the simulations it was fitted
    on, the epochs trained, the span from the record that ends generation to the moment the posterior's record was
    made as the fit ended, the seconds per epoch over that span, the median of the epochs' own seconds, and the
    ``press`` that started it, which the sizes never hold; each None when its source is missing."""
    from core import orchestrator
    noted = _Noted()
    with store_factory() as store, _noting(noted):
        posterior = orchestrator.build_posterior(
            cfg, prior, None, True, store=store, num_runs=num_runs, run_size_cap=run_size_cap,
            hidden_features=hidden_features, num_transforms=num_transforms, max_num_epochs=epochs - 1,
            checkpoint_every=0, resume="auto", fig_sink=fig_sink)
        manifest = getattr(posterior, "manifest", None)
        created = _moment(getattr(manifest, "created", None))
        body = getattr(manifest, "body", None)
        training = body.get("training") if isinstance(body, dict) else None
        trained = _count(training.get("epochs_trained")) if isinstance(training, dict) else None
        del posterior, manifest, body, training     # the record goes with its folder: nothing of it is held on to
    span = None if created is None or noted.generated is None else created - noted.generated
    return ShortFit({"network": [hidden_features, num_transforms], "simulations": noted.simulations,
                     "epochs": trained, "span": span,
                     "per_epoch": span / trained if span is not None and trained else None,
                     "median_epoch": float(statistics.median(noted.took)) if noted.took else None, "press": press})


@dataclass(frozen=True)
class CardCheck:
    """What a card test hands back: its result, names and numbers, which its run's history keeps. It names no record:
    the five it made went with its scratch store."""
    check_result: dict


class _CardStages(QObject):
    """What a card test's run tells of its stages, from the worker's thread: ``stage`` carries each (the engine's name,
    the event, the seconds or None), heard on the thread the object lives on, the window's, as a queued signal is.
    ``tell`` is what the run is handed: it only emits, and never raises."""

    stage = Signal(str, str, object)

    def tell(self, stage, event, seconds) -> None:
        try:
            self.stage.emit(str(stage), str(event), seconds)
        except Exception:                        # noqa: BLE001 -- what tells a run's stages never stops it
            pass


def card_test_seconds(seconds) -> dict:
    """The seconds of each stage a card test ran, as names and numbers: a stage whose seconds are no number is left
    out."""
    held = seconds if isinstance(seconds, dict) else {}
    return {str(stage): float(took) for stage, took in held.items()
            if isinstance(took, (int, float)) and not isinstance(took, bool)}


def card_check(cfg, *, cell: str, t_obs_s: float, seed: int, num_runs: int, run_size_cap: int, n_cal: int,
               epochs: int, hidden_features: int, num_transforms: int, on_stage, fig_sink=None,
               store_factory=scratch_store) -> CardCheck:
    """One card test on ``cfg``, a configuration holding no cell, in a scratch store: the prior at full size, then a
    training on ``num_runs`` batches of at most ``run_size_cap`` simulations of the network ``hidden_features`` by
    ``num_transforms`` for at most ``epochs`` epochs (the limit sent is one less), its calibration on ``n_cal`` test
    datasets and one inference on a simulated cell from ``cell`` over ``t_obs_s`` seconds, all inside one ``seed``; its
    simulations not kept and nothing it makes saved by a name. ``on_stage`` hears each stage begin and end. Its result:
    each stage's seconds, the stages that ran, the share of unusable driven recordings in its training's first total
    (None without one), and how many batches were run in smaller pieces for lack of the card's memory."""
    from core import diagnostics
    noted = _Noted()
    with store_factory() as store, _noting(noted):
        result = diagnostics.card_test(cfg, cell=cell, t_obs_s=t_obs_s, seed=seed, num_runs=num_runs,
                                       run_size_cap=run_size_cap, n_cal=n_cal, max_num_epochs=epochs - 1,
                                       hidden_features=hidden_features, num_transforms=num_transforms,
                                       checkpoint=False, save=False, on_stage=on_stage, fig_sink=fig_sink,
                                       store=store)
        seconds = card_test_seconds(getattr(result, "seconds", None))
        ran = [str(stage) for stage in getattr(result, "stages_run", None) or ()]
        del result                           # its five records go with their folder: nothing of them is held on to
    masked = noted.masked
    return CardCheck({"seconds": seconds, "stages_run": ran,
                      "unusable": masked[0] / masked[1] if masked is not None and masked[1] else None,
                      "split": len(noted.split)})
