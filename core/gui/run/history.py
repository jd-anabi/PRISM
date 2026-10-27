"""The run history: one JSON file per run of the window, ``<run id>.json``, beside the run's text file
``<run id>.txt``, in the folder ``runs`` beside the settings file.

The window alone writes these files, each whole: to ``<run id>.json.tmp`` first, then renamed over the file it
replaces, so a reader meets the last whole file, never half of one, and a temporary file a failed write left is
never read as a run. The engine and the command line never read them.

A file holds every name of ``FIELDS``, in that order, None where its run has nothing:

- ``id``, ``word``, ``name``, ``place``, ``records_root``: which run (``run_id``), the word for it ("Training"), its
  name, the part and page that show it, and the records folder it wrote to: the latest runs and the run a failed
  training can be continued from are read for the records folder in use.
- ``started``, ``ended``, ``last_heard``: local times, written with their offset (a time with no zone is refused).
  ``last_heard`` is the last write of a run that was not seen to end.
- ``outcome`` ("running" until the end, then "finished", "stopped", "failed" or "not seen to end"; "stopped" from
  the moment a stop with a reason of its own is asked, Windows ending the session, so that a process Windows ends
  first leaves it said), ``stop_reason`` (that reason, "Windows shut down"; None for a stop the person asked for, for
  none, and once the run has finished), ``stage``, ``acknowledged``.
- ``tries``: one entry per try, each of ``TRY_FIELDS``: its start and end, how it ended, the kind of failure, and
  the batches and the epochs the run had kept before the try began and after it ended.
- ``plan``, ``settings``, ``memory_limit``: what the run was started with; ``settings`` holds every argument, enough
  to start the same run again.
- ``simulation_set``, of ``SET_FIELDS``: the set's folder, its batches in all, at the start and at the end, the
  simulations per batch and the median seconds per batch.
- ``health``, ``fit``, ``check``, ``result``, ``failure``, ``text``: what the run's page measured, the fit, a
  check's own result, the record the run wrote (its kind, its id, its name and the name it was asked to have, which
  differ when another record took the name while the run was going), how it failed, and its text file's name.

``fit``, of ``FIT_FIELDS``: the epochs, the best epoch and its validation loss, the median seconds per epoch, what
ended the fit by itself ("patience" or "limit"), the network (hidden features, transforms) and the number of
simulations it was fitted on (an epoch's pace is matched by these two), and the kept training: its id, the passes it
holds, whether it was resumed, the processes that trained it, its wall time in seconds and the last save that
failed; then the epochs the fit was expected to run (the estimate, None when there was none) and whether the kept
training was discarded since. After a stop or a failure in the fit, ``epochs`` is the kept training's passes, never
the count of epoch records this run's text holds: the kept training counts every pass saved, a resumed training's
earlier processes included, while the run's records are this run's own.

No torch at import: the one reading that needs it (``continuable``'s, of a simulation set's state) imports it when
it is asked.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .. import settings

FIELDS = ("id", "word", "name", "place", "records_root", "started", "ended", "last_heard", "outcome", "stop_reason",
          "stage", "acknowledged", "tries", "plan", "settings", "memory_limit", "simulation_set", "health", "fit",
          "check", "result", "failure", "text")
#: What ``simulation_set`` holds.
SET_FIELDS = ("folder", "batches", "at_start", "at_end", "per_batch", "seconds_per_batch")
#: What ``fit`` holds.
FIT_FIELDS = ("epochs", "best_epoch", "best_validation_loss", "seconds_per_epoch", "ended_by", "network",
              "simulations", "kept_id", "passes", "resumed", "processes", "wall_seconds", "save_error", "expected",
              "discarded")
#: What each entry of ``tries`` holds.
TRY_FIELDS = ("started", "ended", "outcome", "failure", "batches_before", "batches_after", "epochs_before",
              "epochs_after")
#: How many times in all, and how many seconds apart, a rename refused while another program holds the file open is
#: tried before the refusal is raised.
REPLACE_TRIES = 5
REPLACE_PAUSE_S = 0.2

#: The pause between two tries of a refused rename.
_pause = time.sleep


def runs_folder() -> Path:
    """The folder ``runs`` beside the settings file. Naming it makes nothing: the first file written makes it."""
    return Path(settings.settings().fileName()).parent / "runs"


def run_id(word: str, started: datetime, folder: Path) -> str:
    """A run's id, the stem of its two files: its local start and its word, lower-cased with its spaces as hyphens,
    ``20261008T190211-training``; ``-2``, ``-3`` and on when a file of that stem is in ``folder`` already."""
    stem = f"{started.astimezone():%Y%m%dT%H%M%S}-{word.lower().replace(' ', '-')}"
    folder, candidate, n = Path(folder), stem, 1
    while (folder / f"{candidate}.json").exists() or (folder / f"{candidate}.txt").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def write(folder: Path, record: dict) -> Path:
    """Write ``record`` as its run's history file in ``folder``, which is made when missing, and return the file's
    path. Every name of ``FIELDS`` is written, in that order; a time with its offset. Written whole: to a temporary
    file, made to reach the disk, then renamed over the old file (``_replace``). Raises ValueError, writing nothing,
    for a record with no id, a name the history does not keep, or a time with no zone; an error of the disk is
    raised as it comes, the temporary file removed."""
    _check(record)
    text = json.dumps({name: record.get(name) for name in FIELDS}, default=_time_text, ensure_ascii=False,
                      indent=1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['id']}.json"
    temporary = folder / f"{record['id']}.json.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        _replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def read(path: Path) -> dict | None:
    """The record a history file holds; None for a file that cannot be read whole as one."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def latest(folder: Path, *, records_root: str, n: int) -> list[dict]:
    """The ``n`` newest runs of the records folder ``records_root``, newest first. A run of another records folder
    is not counted, however new."""
    return [record for record in reversed(_records(folder))
            if _same_folder(record.get("records_root"), records_root)][:n]


def continuable(record: dict, *, store, folder: Path) -> int | None:
    """The batch a failed training or narrowing round can be continued from, or None.

    A number only for a run of the records folder in use (``store``'s) that failed or was not seen to end, whose
    simulation set is unfinished and holds a saved batch, while no later run of its word in that records folder has
    started (``folder`` holds the history). The number is the set's own count of saved batches, read from the set's
    folder, and only once the run has ended: the engine rewrites that state as it goes. A run that was stopped is
    not offered here (its own page offers it)."""
    if record.get("outcome") not in ("failed", "not seen to end"):
        return None
    name = (record.get("simulation_set") or {}).get("folder")
    started = _time(record.get("started"))
    root = str(store.root)
    if not name or started is None or not _same_folder(record.get("records_root"), root):
        return None
    for other in _records(folder):
        if (other.get("id") != record.get("id") and other.get("word") == record.get("word")
                and _same_folder(other.get("records_root"), root) and _time(other.get("started")) > started):
            return None
    from core.SBI.training_checkpoint import peek
    state = peek(store.kind_dir("simulation") / name)
    if not state or state.get("complete"):
        return None
    return int(state.get("batches_done") or 0) or None


def estimate(folder: Path, *, word: str, per: str, size: dict) -> tuple[float, str] | None:
    """The pace of the newest ended run of ``word`` and of this size, with its source in words, or None.

    ``per="batch"``: the median seconds per batch of the newest such run with the same simulations per batch
    (``size["per_batch"]``), "the last training of this size, retrain_aug, took 9.7 s per batch". ``per="epoch"``:
    the median seconds per epoch of the newest such run fitted with the same network (``size["network"]``, hidden
    features and transforms) on the same number of simulations (``size["simulations"]``), "... took about 63 minutes
    per epoch". A run with no pace of that kind is passed over."""
    if per not in ("batch", "epoch"):
        raise ValueError(f"a pace is per batch or per epoch, not per {per!r}")
    wanted = ((size.get("per_batch"),) if per == "batch"
              else (_listed(size.get("network")), size.get("simulations")))
    if any(value is None for value in wanted):
        return None
    for record in reversed(_records(folder)):
        if record.get("word") != word or record.get("outcome") in (None, "running"):
            continue
        if per == "batch":
            held = record.get("simulation_set") or {}
            found, seconds = (held.get("per_batch"),), held.get("seconds_per_batch")
        else:
            held = record.get("fit") or {}
            found, seconds = (_listed(held.get("network")), held.get("simulations")), held.get("seconds_per_epoch")
        if found != wanted or not seconds:
            continue
        seconds = float(seconds)
        pace = f"{seconds:.1f} s per batch" if per == "batch" else f"about {_duration(seconds)} per epoch"
        who = f", {record['name']}," if record.get("name") else ""
        return seconds, f"the last {word.lower()} of this size{who} took {pace}"
    return None


def runs_for(folder: Path, simulation_set: str) -> list[dict]:
    """The runs that wrote to the simulation set whose folder is ``simulation_set``, oldest first."""
    return [record for record in _records(folder)
            if (record.get("simulation_set") or {}).get("folder") == simulation_set]


def close_unseen(folder: Path) -> tuple[str, ...]:
    """Rewrite each history file in ``folder`` that still says "running" as a run not seen to end, heard last when
    its file was last written, and return their ids. Only one PRISM runs at a time, so such a file is a run that
    ended unseen; who may call this, and when, is the window's to decide. A run that ended is left as it is."""
    closed = []
    for path in sorted(Path(folder).glob("*.json")):
        record = read(path)
        if record is None or record.get("outcome") != "running" or record.get("id") != path.stem:
            continue
        heard = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        write(folder, {**record, "outcome": "not seen to end", "last_heard": heard})
        closed.append(record["id"])
    return tuple(closed)


def _replace(source: Path, target: Path) -> None:
    """Rename ``source`` over ``target``. Windows refuses the rename (``PermissionError``) while another program, a
    scanner, an indexer or an editor, holds ``target`` open, mostly for a moment: such a refusal is tried again,
    ``REPLACE_TRIES`` times in all, ``REPLACE_PAUSE_S`` seconds apart, and the last one raised. Any other error is
    raised at once."""
    for tried in range(1, REPLACE_TRIES + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if tried == REPLACE_TRIES:
                raise
            _pause(REPLACE_PAUSE_S)


def _records(folder: Path) -> list[dict]:
    """Every run's record in ``folder``, oldest start first. A file that is not a whole record with a start is
    passed over, and a temporary one is never looked at."""
    found = []
    for path in sorted(Path(folder).glob("*.json")):
        record = read(path)
        started = _time(record.get("started")) if record is not None else None
        if started is not None:
            found.append((started, record))
    return [record for _started, record in sorted(found, key=lambda pair: pair[0])]


def _check(record: dict) -> None:
    """Refuse, before anything is written, a record with no id or with a name the history does not keep."""
    if not record.get("id"):
        raise ValueError("a run's history file needs the run's id")
    parts = [("the record", FIELDS, record), ("simulation_set", SET_FIELDS, record.get("simulation_set")),
             ("fit", FIT_FIELDS, record.get("fit"))]
    parts += [("a try", TRY_FIELDS, one) for one in record.get("tries") or ()]
    for what, kept, held in parts:
        unknown = sorted(set(held) - set(kept)) if isinstance(held, dict) else []
        if unknown:
            raise ValueError(f"{what} holds what a run's history does not keep: {unknown}")


def _time_text(value):
    """A time as the history writes it: local, with its offset. A time with no zone is refused."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"a time with no zone cannot go into a run's history: {value!r}")
        return value.isoformat()
    raise TypeError(f"a {type(value).__name__} cannot go into a run's history")


def _time(value) -> datetime | None:
    """A time of a record, as the history wrote it (text) or as it was handed to ``write`` (a time); None for
    anything else, a time with no zone among it."""
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
    return moment if moment.tzinfo is not None else None


def _same_folder(one, other) -> bool:
    """Whether two spellings name the same folder."""
    if not one or not other:
        return False
    return os.path.normcase(os.path.abspath(one)) == os.path.normcase(os.path.abspath(other))


def _listed(value):
    """A network's two numbers as one comparable value, however they were held."""
    return list(value) if isinstance(value, (list, tuple)) else value


def _duration(seconds: float) -> str:
    """A pace's time in words: whole seconds under a minute, else whole minutes."""
    if seconds < 59.5:
        count, unit = round(seconds), "second"
    else:
        count, unit = round(seconds / 60), "minute"
    return f"{count} {unit}{'' if count == 1 else 's'}"
