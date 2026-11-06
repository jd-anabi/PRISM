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
  none, and once the run has finished), ``stage`` (the name of the stage the run was in last), ``stages`` (the run's
  stages, each its name, the readings that begin it and what counts it, so that its page in the past view draws them
  from its text), ``acknowledged``.
- ``tries``: one entry per try, each of ``TRY_FIELDS``: its start and end, how it ended, the kind of failure, and
  the batches and the epochs the run had kept before the try began and after it ended. The try a run is in as
  Windows ends the session is written "stopped" with the run, from the moment that stop is asked.
- ``plan``, ``settings``, ``memory_limit``: what the run was started with. ``plan`` and ``settings`` are what its page
  hands, made plain (``plain``), None until it hands them; ``settings`` is for every argument, enough to start the
  same run again.
  ``memory_limit`` is ``{"gib", "source", "computer"}``: the limit in force for each batch of simulations (None for
  none), where it was set ("setting", "environment" or "none") and the computer the run ran on. A file written before
  these were kept holds a bare number or None, which names no computer.
- ``simulation_set``, of ``SET_FIELDS``: the set's folder, its batches in all, at the start and at the end, the
  simulations per batch and the median seconds per batch.
- ``health``, ``fit``, ``check``, ``result``, ``failure``, ``text``: what the run's page measured, the fit, a
  check's own result, the record the run wrote (its kind, its id, its name and the name it was asked to have, which
  differ when another record took the name while the run was going), how it failed, and its text file's name.
  ``health`` is ``{"failed", "simulated", "memory_peak", "batches_split", "out_of_memory"}`` once any of it is known,
  else None; ``check`` is None until a check hands one over. A file written before these were kept holds the first two
  alone. A name a reader finds missing or None, in a file of any age, is no information.

``fit``, of ``FIT_FIELDS``: the epochs, the best epoch and its validation loss, the median seconds per epoch, what
ended the fit by itself ("patience" or "limit"), the network (hidden features, transforms) and the number of
simulations it was fitted on (an epoch's pace is matched by these two), and the kept training: its id, the passes it
holds, whether it was resumed, the processes that trained it, its wall time in seconds and the last save that
failed; then the epochs the fit was expected to run (the estimate, None when there was none) and whether the kept
training was discarded since. After a stop or a failure in the fit, ``epochs`` is the kept training's passes, never
the count of epoch records this run's text holds: the kept training counts every pass saved, a resumed training's
earlier processes included, while the run's records are this run's own.

A file that cannot be read whole, or that holds a name in a shape no run's history holds it (``SHAPES``), is no
run's history the window can read: every reader passes it over, as though it were not there, and the window's log
says which file, once.

No torch: what a simulation set holds is read by the window's reading of what a reopened window can offer, never
here.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from .. import settings
from ..texts import is_size

log = logging.getLogger(__name__)

FIELDS = ("id", "word", "name", "place", "records_root", "started", "ended", "last_heard", "outcome", "stop_reason",
          "stage", "stages", "acknowledged", "tries", "plan", "settings", "memory_limit", "simulation_set", "health",
          "fit", "check", "result", "failure", "text")
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
#: How deep ``plain`` follows a value before it says the rest in words: this bounds how deep, not how much. Nothing a
#: page hands goes so deep, and a value that holds itself goes on for ever.
PLAIN_DEPTH = 16
#: What a name of a history holds when it holds anything but None, as the window writes it, for each name a reader
#: takes as such; a place's two parts and each try are of their own shape too.
SHAPES = {"id": str, "place": list, "records_root": str, "outcome": str, "stage": str, "stages": list,
          "acknowledged": bool, "tries": list, "simulation_set": dict, "health": dict, "fit": dict}
#: The window's log, the first time a file in the folder of runs is passed over as no run's history it can read:
#: the file's path and why.
PASSED_OVER_TEXT = "The file %s is no run's history the window can read (%s); it is passed over."

#: The pause between two tries of a refused rename.
_pause = time.sleep
#: The files already said to be passed over, each by its path, its size and when it was last written: a file written
#: again is said again.
_said: set = set()


def runs_folder() -> Path:
    """The folder ``runs`` beside the settings file. Naming it makes nothing: the first file written makes it."""
    return Path(settings.settings().fileName()).parent / "runs"


def run_id(word: str, started: datetime, folder: Path | None) -> str:
    """A run's id, the stem of its two files: its local start and its word, lower-cased with its spaces as hyphens,
    ``20261008T190211-training``; ``-2``, ``-3`` and on when a file of that stem is in ``folder`` already. With no
    folder (None), the stem alone, no folder read: the id of a run that writes no file, when its folder cannot be
    read."""
    stem = f"{started.astimezone():%Y%m%dT%H%M%S}-{word.lower().replace(' ', '-')}"
    if folder is None:
        return stem
    folder, candidate, n = Path(folder), stem, 1
    while (folder / f"{candidate}.json").exists() or (folder / f"{candidate}.txt").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def write(folder: Path, record: dict) -> Path:
    """Write ``record`` as its run's history file in ``folder``, which is made when missing, and return the file's
    path. Every name of ``FIELDS`` is written, in that order; a time with its offset. Written whole: to a temporary
    file, made to reach the disk, then renamed over the old file (``_replace``). A character no encoding can hold, a
    lone surrogate, is written as its escape, and reads back as it was. Raises ValueError, writing nothing, for a record
    with no id, a name the history does not keep, or a time with no zone; an error of the disk is raised as it comes,
    the temporary file removed."""
    _check(record)
    text = json.dumps({name: record.get(name) for name in FIELDS}, default=_time_text, ensure_ascii=False,
                      indent=1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['id']}.json"
    temporary = folder / f"{record['id']}.json.tmp"
    try:
        with open(temporary, "w", encoding="utf-8", errors="backslashreplace") as file:
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


def plain(value, depth: int = 0):
    """``value`` as a history can hold it: what JSON holds as it is; a mapping with text for its keys; a sequence or a
    set as a list; an array or an array's number as plain numbers; a time as its text; anything else in words, a whole
    number too long to write among it. A new value each time, none of it the one handed, and plain of plain is the same.
    ``write`` raises for a whole record over one value it cannot hold, so what a page hands a run for its history comes
    through here at the run's start, and nothing here raises: what cannot be gone through is said in words."""
    if value is None or isinstance(value, (bool, float, str)):
        return value
    try:
        if isinstance(value, int):
            int.__repr__(value)                  # what the writer asks of a whole number, which past a limit raises
            return value
        if depth < PLAIN_DEPTH:
            if isinstance(value, dict):
                return {_words(key): plain(held, depth + 1) for key, held in value.items()}
            if isinstance(value, (list, tuple, set, frozenset)):
                return [plain(held, depth + 1) for held in value]
            if hasattr(value, "tolist"):
                return plain(value.tolist(), depth + 1)
        if isinstance(value, datetime):
            return value.isoformat()
    except Exception:                            # noqa: BLE001 -- a run's start never fails over its history
        pass
    return _words(value)


def _words(value) -> str:
    """``value`` in words, or by its kind when it cannot say what it is."""
    try:
        return str(value)
    except Exception:                            # noqa: BLE001 -- a run's start never fails over its history
        return "a whole number too long to write" if isinstance(value, int) else f"a {type(value).__name__}"


def read(path: Path) -> dict | None:
    """The record a history file holds; None for a file that is not there, for one that cannot be read whole as one,
    and for one that holds a name in a shape no run's history holds it (``SHAPES``), so that every reader passes such
    a file over as though it were not there; the window's log names such a file, once for each time it was
    written."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        _passed_over(path, f"{type(error).__name__}: {error}")
        return None
    wrong = _misshapen(record)
    if wrong:
        _passed_over(path, wrong)
        return None
    return record


def _misshapen(record) -> str:
    """What makes ``record`` no run's history the window can read, in words; "" when nothing does."""
    if not isinstance(record, dict):
        return "it holds no record"
    for name, shape in SHAPES.items():
        value = record.get(name)
        if value is not None and not isinstance(value, shape):
            return f"its {name} should hold {shape.__name__}, not {type(value).__name__}"
    if any(part is not None and not isinstance(part, str) for part in record.get("place") or ()):
        return "its place holds what is no name"
    if any(not isinstance(one, dict) for one in record.get("tries") or ()):
        return "a try is not a record"
    return ""


def _passed_over(path: Path, why: str) -> None:
    """The window's log says that the file at ``path`` is passed over, and why: once for each time it was written."""
    try:
        written = path.stat()
        key = (str(path), written.st_size, written.st_mtime_ns)
    except OSError:
        key = (str(path), None, None)
    if key not in _said:
        _said.add(key)
        log.warning(PASSED_OVER_TEXT, path, why)


def latest(folder: Path, *, records_root: str, n: int) -> list[dict]:
    """The ``n`` newest runs of the records folder ``records_root``, newest first. A run of another records folder
    is not counted, however new."""
    return [record for record in reversed(_records(folder))
            if _same_folder(record.get("records_root"), records_root)][:n]


def newest_unseen(folder: Path, *, records_root: str) -> dict | None:
    """The newest run of the records folder ``records_root`` that has ended (or was not seen to end) and whose end
    nobody has seen, or None. A run still marked "running" has not ended, and is passed over."""
    for record in reversed(_records(folder)):
        if (_same_folder(record.get("records_root"), records_root) and not record.get("acknowledged")
                and record.get("outcome") in ("finished", "stopped", "failed", "not seen to end")):
            return record
    return None


def runs_of(folder: Path, *, records_root: str) -> list[dict]:
    """Every run of the records folder ``records_root``, oldest start first: what a reopened window can offer is worked
    out from them (``core/gui/continuing.py``). A run of another records folder is not counted."""
    return [record for record in _records(folder) if _same_folder(record.get("records_root"), records_root)]


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


def duration(folder: Path, word: str) -> tuple[float, str] | None:
    """How long the newest finished run of ``word`` took, in seconds, and its name ("" for none); None when no such
    run says both its times. Every run in ``folder`` counts, whichever records it wrote to: this says what this
    computer took."""
    for record in reversed(_records(folder)):
        started, ended = _time(record.get("started")), _time(record.get("ended"))
        if record.get("word") == word and record.get("outcome") == "finished" and started and ended:
            return (ended - started).total_seconds(), str(record.get("name") or "")
    return None


def last_on(folder: Path, computer: str) -> dict | None:
    """The newest run in ``folder`` whose memory limit names the computer ``computer``, or None. A file that names no
    computer, as one written before runs named theirs, is passed over, and every file is when ``computer`` is ""."""
    if not computer:
        return None
    for record in reversed(_records(folder)):
        limit = record.get("memory_limit")
        if isinstance(limit, dict) and limit.get("computer") == computer:
            return record
    return None


def last_measured_on(folder: Path, computer: str) -> dict | None:
    """The newest run in ``folder`` of the computer ``computer`` that measured its batches, or None."""
    if not computer:
        return None
    for record in reversed(_records(folder)):
        limit, health = record.get("memory_limit"), record.get("health") or {}
        if isinstance(limit, dict) and limit.get("computer") == computer and is_size(health.get("memory_peak")):
            return record
    return None


def runs_for(folder: Path, simulation_set: str) -> list[dict]:
    """The runs that wrote to the simulation set whose folder is ``simulation_set``, oldest first."""
    return [record for record in _records(folder)
            if (record.get("simulation_set") or {}).get("folder") == simulation_set]


def close_unseen(folder: Path) -> tuple[str, ...]:
    """Rewrite each history file in ``folder`` that still says "running" as a run not seen to end, heard last when
    its file was last written, and return their ids. Only one PRISM runs at a time, so such a file is a run that
    ended unseen; who may call this, and when, is the window's to decide. A run that ended is left as it is, and so is
    a file that cannot be read or written again, which the window's log names."""
    closed = []
    for path in sorted(Path(folder).glob("*.json")):
        record = read(path)
        if record is None or record.get("outcome") != "running" or record.get("id") != path.stem:
            continue
        try:
            heard = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            write(folder, {**record, "outcome": "not seen to end", "last_heard": heard})
        except (OSError, ValueError) as error:
            _passed_over(path, f"{type(error).__name__}: {error}")
            continue
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
